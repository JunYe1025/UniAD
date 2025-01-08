#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner.base_module import BaseModule
from projects.mmdet3d_plugin.models.utils.functional import (
    norm_points,
    pos2posemb2d,
    trajectory_coordinate_transform
)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MotionTransformerDecoder(BaseModule): # 这个类是MotionFormer的解码器，用于生成预测的轨迹。这部分MotionTransformerDecoder其实相当于整个Figure 4
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, pc_range=None, embed_dims=256, transformerlayers=None, num_layers=3, **kwargs):
        super(MotionTransformerDecoder, self).__init__()
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.intention_interaction_layers = IntentionInteraction()
        self.track_agent_interaction_layers = nn.ModuleList(
            [TrackAgentInteraction() for i in range(self.num_layers)]) # 3个, Figure 4中的N
        self.map_interaction_layers = nn.ModuleList(
            [MapInteraction() for i in range(self.num_layers)])
        self.bev_interaction_layers = nn.ModuleList(
            [build_transformer_layer(transformerlayers) for i in range(self.num_layers)])

        self.static_dynamic_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.dynamic_embed_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*3, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.in_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.out_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*4, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

    def forward(self,
                track_query,
                lane_query,
                track_query_pos=None,
                lane_query_pos=None,
                track_bbox_results=None,
                bev_embed=None,
                reference_trajs=None,
                traj_reg_branches=None,
                agent_level_embedding=None,
                scene_level_ego_embedding=None,
                scene_level_offset_embedding=None,
                learnable_embed=None,
                agent_level_embedding_layer=None,
                scene_level_ego_embedding_layer=None,
                scene_level_offset_embedding_layer=None,
                **kwargs):
        """Forward function for `MotionTransformerDecoder`.
        Args:
            agent_query (B, A, D) # Q_A  = N_a × 256; agent features from TrackFormer
            map_query (B, M, D)  # Q_M = N_m × 256; map features from MapFormer (地图查询，其中 M 表示地图中的对象数量)
            map_query_pos (B, G, D) # 地图查询位置。
            static_intention_embed (B, A, P, D) # 静态意图嵌入，其中 P 表示意图的数量。代表每个代理的静态或固定的意图。
            offset_query_embed (B, A, P, D) # 偏移查询嵌入，与意图的偏移或变化有关。
            global_intention_embed (B, A, P, D) # 全局意图嵌入，代表每个代理的全局意图。
            learnable_intention_embed (B, A, P, D) # 可学习意图嵌入，是模型在训练过程中学习的意图表示。
            det_query_pos (B, A, D) # 检测查询位置，用于将查询嵌入与检测位置对齐。检测查询位置，代表与检测任务相关的位置信息
        Returns:
            None
        """

        # ***这里意图（intention）代表目标位置***

        intermediate = []
        intermediate_reference_trajs = []

        B, _, P, D = agent_level_embedding.shape
        track_query_bc = track_query.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)
        track_query_pos_bc = track_query_pos.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)

        # static intention embedding, which is imutable throughout all layers
        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)

        # static_intention_embed 是静态意图嵌入，表示每个代理的固定意图(感觉相当于Figure 4底下那几个量？？？)。
        # 然后这几个量是不会变的（这些意图在整个网络的层次中保持不变），所以用了static，通常用于捕捉代理（agent）的基本行为模式或目标位置。
        static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed # learnable_embed是什么？？？
        reference_trajs_input = reference_trajs.unsqueeze(4).detach()

        query_embed = torch.zeros_like(static_intention_embed)
        for lid in range(self.num_layers):
            # fuse static and dynamic intention embedding
            # the dynamic intention embedding is the output of the previous layer, which is initialized with anchor embedding
            # dynamic_query_embed 是动态查询嵌入，表示在每一层中计算出的代理的当前意图。这些嵌入会随着网络的前向传播而更新，反映出代理在不同时间步长或上下文中的变化。
            dynamic_query_embed = self.dynamic_embed_fuser(torch.cat(
                [agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1))
            
            # fuse static and dynamic intention embedding
            query_embed_intention = self.static_dynamic_fuser(torch.cat(
                [static_intention_embed, dynamic_query_embed], dim=-1))  # (B, A, P, D)
            
            # fuse intention embedding with query embedding
            # 这里的query_embed相当于Figure 4中的Q，其包括static和dynamic两个部分，然后也会考虑之前的query_embed
            query_embed = self.in_query_fuser(torch.cat([query_embed, query_embed_intention], dim=-1)) 
            
            # interaction between agents
            # 这个track_query_embed是论文中的Q_A吧
            track_query_embed = self.track_agent_interaction_layers[lid](
                query_embed, track_query, query_pos=track_query_pos_bc, key_pos=track_query_pos)
            
            # interaction between agents and map
            # 这个lane_query是论文中的Q_M吧
            map_query_embed = self.map_interaction_layers[lid](
                query_embed, lane_query, query_pos=track_query_pos_bc, key_pos=lane_query_pos)
            
            # 代理与目标（BEV，即鸟瞰图）之间的交互，使用可变形Transformer实现 
            # interaction between agents and bev, ie. interaction between agents and goals
            # implemented with deformable transformer
            bev_query_embed = self.bev_interaction_layers[lid](
                query_embed,
                value=bev_embed,
                query_pos=track_query_pos_bc,
                bbox_results=track_bbox_results,
                reference_trajs=reference_trajs_input,
                **kwargs)
            
            # fusing the embeddings from different interaction layers
            # 这里的query_embed相当于Figure 4中的Q_a、Q_m、Q_g
            query_embed = [track_query_embed, map_query_embed, bev_query_embed, track_query_bc+track_query_pos_bc]
            query_embed = torch.cat(query_embed, dim=-1)
            query_embed = self.out_query_fuser(query_embed)

            if traj_reg_branches is not None: #traj_reg_branches相当于Figure 4中虚线框内最左边的那个MLP
                # update reference trajectory
                tmp = traj_reg_branches[lid](query_embed)
                bs, n_agent, n_modes, n_steps, _ = reference_trajs.shape # 最后一个维度是（Dimensionality of each trajectory point = 2）
                tmp = tmp.view(bs, n_agent, n_modes, n_steps, -1)
                
                # we predict speed of trajectory and use cumsum trick to get the trajectory
                # 预测的是每个时间步的速度增量（而不是直接预测位置）【这是一个常用的轨迹预测技巧，因为预测速度增量比直接预测位置更容易学习】
                # tmp[..., :2] 等价于 tmp[:, :, :, :, :2]；dim=3表示在第4个维度（时间步维度）上进行累积求和；如果预测值是速度增量：[Δv1, Δv2, Δv3, ...]，那么累计后就是[Δv1, Δv1+Δv2, Δv1+Δv2+Δv3, ...]
                tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3) 
                new_reference_trajs = torch.zeros_like(reference_trajs)
                new_reference_trajs = tmp[..., :2]
                reference_trajs = new_reference_trajs.detach()
                reference_trajs_input = reference_trajs.unsqueeze(4)  # BS NUM_AGENT NUM_MODE 12 NUM_LEVEL  2 （这里的NUM_LEVEL（值为1）是为了与transformer层数对齐的一个维度）

                # update embedding, which is used in the next layer
                # only update the embedding of the last step, i.e. the goal
                ep_offset_embed = reference_trajs.detach() # 全局坐标系下的轨迹
                ep_ego_embed = trajectory_coordinate_transform(reference_trajs.unsqueeze(
                    2), track_bbox_results, with_translation_transform=True, with_rotation_transform=False).squeeze(2).detach() # 自车坐标系下的轨迹（只进行平移变换）
                ep_agent_embed = trajectory_coordinate_transform(reference_trajs.unsqueeze(
                    2), track_bbox_results, with_translation_transform=False, with_rotation_transform=True).squeeze(2).detach() # 智能体坐标系下的轨迹（只进行旋转变换）


                # ep_agent_embed[..., -1, :] 表示最后一个时间步的智能体坐标系下的轨迹（点）
                # norm_points(points, pc_range)将点坐标归一化到[0,1]范围，pc_range定义了点云的范围
                # pos2posemb2d(normalized_points)将归一化的2D位置转换为位置嵌入（通常使用正弦位置编码）
                # agent_level_embedding_layer(position_embedding)，最后通过MLP网络处理位置嵌入（从base_motion_head.py可以看到这是一个两层的MLP）
                agent_level_embedding = agent_level_embedding_layer(pos2posemb2d(
                    norm_points(ep_agent_embed[..., -1, :], self.pc_range))) #这整个过程的目的是将轨迹终点的空间位置信息转换为适合神经网络处理的特征表示。
                # 初始输入pos: [bs, n_agent, n_modes, 2], 即ep_agent_embed[..., -1, :]的维度是[bs, n_agent, n_modes, 2]
                # pos2posemb2d处理后: [bs, n_agent, n_modes, 256]  (x和y各128维拼接)【可以看出pos2posemb2d是将2D位置特征转换为了256维的位置嵌入(向量)】
                # 经过agent_level_embedding_layer(MLP)后: [bs, n_agent, n_modes, 256]
                
                scene_level_ego_embedding = scene_level_ego_embedding_layer(pos2posemb2d(
                    norm_points(ep_ego_embed[..., -1, :], self.pc_range)))
                scene_level_offset_embedding = scene_level_offset_embedding_layer(pos2posemb2d(
                    norm_points(ep_offset_embed[..., -1, :], self.pc_range)))

                # 保存每一层的预测结果，用于计算辅助损失
                intermediate.append(query_embed)
                intermediate_reference_trajs.append(reference_trajs)

        return torch.stack(intermediate), torch.stack(intermediate_reference_trajs)


class TrackAgentInteraction(BaseModule):
    """
    Modeling the interaction between the agents
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
                                                                  batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None):
        '''
        query: context query (B, A, P, D) 
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        mem = key.expand(B*A, -1, -1)
        # N, A, P, D -> N*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query #这个query相当于论文中的Q_a


class MapInteraction(BaseModule):
    """
    Modeling the interaction between the agent and the map
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
                                                                  batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None):
        '''
        x: context query (B, A, P, D) 
        query_pos: mode pos embedding (B, A, P, D)
        '''

        # B (Batch): 批次大小 ; A (Agent): 场景中的智能体(车辆)数量
        # P: 每个智能体的预测轨迹数量 = number of forecasting modality in MotionFormer = 6 ;
        # D (Dimension): 每个位置的特征维度 = 256
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        
        # N, A, P, D -> N*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)

        # Key相当于论文中的Q_M经过了（PE+MLP）得到的K，即map features from MapFormer，维度为N_m × 256 （300 x 256）
        mem = key.expand(B*A, -1, -1)  # 将地图特征从 [B, M, D] 扩展为 [B*A, M, D]
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query #这个query相当于论文中的Q_m


class IntentionInteraction(BaseModule):
    """
    Modeling the interaction between anchors
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerEncoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
                                                                  batch_first=batch_first)

    def forward(self, query):
        B, A, P, D = query.shape
        # B, A, P, D -> B*A,P, D
        rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = rebatch_x.view(B, A, P, D)
        return out
