import torch
import torch.nn as nn
from mmdet.models.builder import HEADS, build_loss
from einops import rearrange
from projects.mmdet3d_plugin.models.utils.functional import bivariate_gaussian_activation
from .planning_head_plugin import CollisionNonlinearOptimizer
import numpy as np

@HEADS.register_module()
class PlanningHeadSingleMode(nn.Module):
    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 embed_dims=256,
                 planning_steps=6,
                 loss_planning=None,
                 loss_collision=None,
                 planning_eval=False,
                 use_col_optim=False,
                 col_optim_args=dict(
                    occ_filter_range=5.0,
                    sigma=1.0, 
                    alpha_collision=5.0,
                 ),
                ):
        """
        Single Mode Planning Head for Autonomous Driving.

        Args:
            embed_dims (int): Embedding dimensions. Default: 256.
            planning_steps (int): Number of steps for motion planning. Default: 6.
            loss_planning (dict): Configuration for planning loss. Default: None.
            loss_collision (dict): Configuration for collision loss. Default: None.
            planning_eval (bool): Whether to use planning for evaluation. Default: False.
            use_col_optim (bool): Whether to use collision optimization. Default: False.
            col_optim_args (dict): Collision optimization arguments. Default: dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0).
        """
        super(PlanningHeadSingleMode, self).__init__()

        # Nuscenes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.navi_embed = nn.Embedding(3, embed_dims)
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, planning_steps * 2),
        )
        self.loss_planning = build_loss(loss_planning)
        self.planning_steps = planning_steps
        self.planning_eval = planning_eval
        
        #### planning head
        fuser_dim = 3
        attn_module_layer = nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)
        
        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*fuser_dim, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        
        self.pos_embed = nn.Embedding(1, embed_dims)
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        self.loss_collision = nn.ModuleList(self.loss_collision)

        self.use_col_optim = use_col_optim
        self.occ_filter_range = col_optim_args['occ_filter_range']
        self.sigma = col_optim_args['sigma']
        self.alpha_collision = col_optim_args['alpha_collision']

    def forward_train(self,
                      bev_embed, 
                      outs_motion={}, 
                      outs_occflow={},
                      sdc_planning=None, 
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      img_metas=None,
                      ):
        """
        Perform forward planning training with the given inputs.
        Args:
            bev_embed (torch.Tensor): The input bird's eye view feature map.
            outs_motion (dict): A dictionary containing the motion outputs.
            outs_occflow (dict): A dictionary containing the occupancy flow outputs.
            sdc_planning (torch.Tensor, optional): The self-driving car's planned trajectory.
            sdc_planning_mask (torch.Tensor, optional): The mask for the self-driving car's planning.
            command (torch.Tensor, optional): The driving command issued to the self-driving car.
            gt_future_boxes (torch.Tensor, optional): The ground truth future bounding boxes.
            img_metas (list[dict], optional): A list of metadata information about the input images.

        Returns:
            ret_dict (dict): A dictionary containing the losses and planning outputs.
        """
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']

        occflow_ins_mask = outs_occflow.get('pred_ins_masks', None)
        
        outs_planning = self(bev_embed, occflow_ins_mask, bev_pos, sdc_traj_query, sdc_track_query, command)
        loss_inputs = [sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes]
        losses = self.loss(*loss_inputs, img_metas=img_metas)
        ret_dict = dict(losses=losses, outs_motion=outs_planning)
        return ret_dict

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={}, command=None):
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']
        occflow_ins_mask = outs_occflow.get('pred_ins_masks', None)
        
        outs_planning = self(bev_embed, occflow_ins_mask, bev_pos, sdc_traj_query, sdc_track_query, 
                                            command)
        return outs_planning

    def forward(self, 
                bev_embed, 
                occflow_ins_mask, 
                bev_pos, 
                sdc_traj_query, 
                sdc_track_query, 
                command):
        """
        Forward pass for PlanningHeadSingleMode.

        Args:
            bev_embed (torch.Tensor): Bird's eye view feature embedding.
            occflow_ins_mask (torch.Tensor): Instance mask for occupancy flow.
            bev_pos (torch.Tensor): BEV position.
            sdc_traj_query (torch.Tensor): SDC trajectory query.
            sdc_track_query (torch.Tensor): SDC track query.
            command (int): Driving command.

        Returns:
            dict: A dictionary containing SDC trajectory and all SDC trajectories.
        """
        sdc_track_query = sdc_track_query.detach()
        sdc_traj_query = sdc_traj_query[-1]
        P = sdc_traj_query.shape[1]
        sdc_track_query = sdc_track_query[:, None].expand(-1,P,-1)
        
        
        navi_embed = self.navi_embed.weight[command]
        navi_embed = navi_embed[None].expand(-1,P,-1)
        plan_query = torch.cat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)

        plan_query = self.mlp_fuser(plan_query).max(1, keepdim=True)[0]   # expand, then fuse  # [1, 6, 768] -> [1, 1, 256]
        plan_query = rearrange(plan_query, 'b p c -> p b c')
        # bev_embed: [40000, 1, 256]
        
        bev_pos = rearrange(bev_pos, 'b c h w -> (h w) b c')
        bev_feat = bev_embed +  bev_pos  # [40000, 1, 256]
      
        pos_embed = self.pos_embed.weight
        plan_query = plan_query + pos_embed[None]  # [1, 1, 256]
        
        # plan_query: [1, 1, 256]
        # bev_feat: [40000, 1, 256]
        plan_query = self.attn_module(plan_query, bev_feat)   # [1, 1, 256]
        
        sdc_traj_all = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))
        sdc_traj_all[...,:2] = torch.cumsum(sdc_traj_all[...,:2], dim=2)
        sdc_traj_all[0] = bivariate_gaussian_activation(sdc_traj_all[0])
        if self.use_col_optim and not self.training:
            # post process, only used when testing
            sdc_traj_all = self.collision_optimization(sdc_traj_all, occflow_ins_mask)
        
        return dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
        )

    def collision_optimization(self, sdc_traj_all, occflow_ins_mask):
        """
        Optimize SDC trajectory with occupancy instance mask.

        Args:
            sdc_traj_all (torch.Tensor): SDC trajectory tensor.
            occflow_ins_mask (torch.Tensor): Occupancy flow instance mask. 
        Returns:
            torch.Tensor: Optimized SDC trajectory tensor.
        """
        if occflow_ins_mask is None:
            return sdc_traj_all
        occflow_ins_mask = occflow_ins_mask.sigmoid()
        pos_xy_t = []
        valid_occupancy_num = 0
        for t in range(self.planning_steps):  
            pos_xy = torch.nonzero(occflow_ins_mask[0].sum(0)[min(t+1, 4)] > 0.1, as_tuple=False)
            pos_xy = pos_xy[:, [1, 0]]
            pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w//2) * 0.5 + 0.25

            # filter the occupancy in range
            keep_index = torch.sum((sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2])**2, axis=-1) < self.occ_filter_range**2
            pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())
            valid_occupancy_num += torch.sum(keep_index>0)
        if valid_occupancy_num == 0:
            return sdc_traj_all
        
        col_optimizer = CollisionNonlinearOptimizer(self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t)
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())
        sol = col_optimizer.solve()
        sdc_traj_optim = np.stack([sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)], axis=-1)
        return torch.tensor(sdc_traj_optim[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype)
    
    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, future_gt_bbox=None, img_metas=None):
        sdc_traj_all = outs_planning['sdc_traj_all'] # b, p, t, 5
        loss_dict = dict()
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :3], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1), future_gt_bbox[0][1:self.planning_steps+1])
            loss_dict[f'loss_collision_{i}'] = loss_collision          
        loss_ade = self.loss_planning(sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :2], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1))
        loss_dict.update(dict(loss_ade=loss_ade))
        return loss_dict