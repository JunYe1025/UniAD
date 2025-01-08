import math
import torch
from einops import rearrange, repeat

def bivariate_gaussian_activation(ip):
    """
    Activation function to output parameters of bivariate Gaussian distribution.

    Args:
        ip (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor containing the parameters of the bivariate Gaussian distribution.
    """

    """
    将轨迹预测转换为二元高斯分布参数
    输入: [batch_size, num_modes, steps, 5]  # (x, y, σx, σy, ρ)
    输出: 每个时间步的位置分布

    作用：将确定性的轨迹点转换为概率分布，每个预测点用二维高斯分布表示，可以表达预测的不确定性，有助于计算概率性的损失函数
    """

    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out

def norm_points(pos, pc_range):
    """
    Normalize the end points of a given position tensor.

    Args:
        pos (torch.Tensor): Input position tensor.
        pc_range (List[float]): Point cloud range.

    Returns:
        torch.Tensor: Normalized end points tensor.
    """
    x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1]) 
    return torch.stack([x_norm, y_norm], dim=-1)

def pos2posemb2d(pos, num_pos_feats=128, temperature=10000): # pos2posemb2d的作用是将2D位置特征转换为256维的嵌入向量
    """
    Convert 2D position into positional embeddings.

    Args:
        pos (torch.Tensor): Input 2D position tensor.
        num_pos_feats (int, optional): Number of positional features. Default is 128.
        temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.

    Returns:
        torch.Tensor: Positional embeddings tensor.
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device) # [start, end)。也就是[0，1，...，127]
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats) # //表示整数除法，也就是dim_t除以2得到的整数部分
    # 让我们分解这个计算：
    # 1. dim_t // 2: [0,0,1,1,2,2,...,63,63]
    # 2. 2 * (dim_t // 2): [0,0,2,2,4,4,...,126,126]
    # 3. ... / num_pos_feats: [0,0,2/128,2/128,4/128,4/128,...,126/128,126/128]
    # 4. temperature ** (...): [1,1,10000^(2/128),10000^(2/128),...,10000^(126/128),10000^(126/128)] 
    # 这个操作创建了一个几何级数序列，用于后续的位置编码。这种设计使得不同维度能够捕获不同尺度的位置信息。


    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    # 假设输入pos的维度是 [bs, n_agent, n_modes, 2]
    # 1. pos[..., 0]: 选择x坐标
    # 维度变为: [bs, n_agent, n_modes]
    # 2. pos[..., 0, None]: 增加一个维度
    # 维度变为: [bs, n_agent, n_modes, 1]
    # 3. dim_t的维度是 [128]
    # 4. pos[..., 0, None] / dim_t: 广播除法
    # 最终维度变为: [bs, n_agent, n_modes, 128]

    # 0::2 --> start:stop（省略了）:step
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2) # 维度: [bs, n_agent, n_modes, 64, 2]
    # 1. pos_x[..., 0::2]选择偶数索引
    # 维度: [bs, n_agent, n_modes, 64]
    # 2. pos_x[..., 1::2]选择奇数索引
    # 维度: [bs, n_agent, n_modes, 64]
    # 3. torch.stack(..., dim=-1)
    # pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1) --> 维度: [bs, n_agent, n_modes, 64, 2]
    # 4. flatten(-2)：将最后两个维度展平
    # pos_x = pos_x.flatten(-2) --> 维度: [bs, n_agent, n_modes, 128]
    
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1) # [bs, n_agent, n_modes, 256]
    return posemb

def rot_2d(yaw):
    """
    Compute 2D rotation matrix for a given yaw angle tensor.

    Args:
        yaw (torch.Tensor): Input yaw angle tensor.

    Returns:
        torch.Tensor: 2D rotation matrix tensor.
    """
    sy, cy = torch.sin(yaw), torch.cos(yaw)
    out = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).permute([2,0,1])
    return out

def anchor_coordinate_transform(anchors, bbox_results, with_translation_transform=True, with_rotation_transform=True):
    """
    Transform anchor coordinates with respect to detected bounding boxes in the batch.

    Args:
        anchors (torch.Tensor): A tensor containing the k-means anchor values.
        bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
        with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
        with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.

    Returns:
        torch.Tensor: A tensor containing the transformed anchor coordinates.
    """
    batch_size = len(bbox_results)
    batched_anchors = []
    transformed_anchors = anchors[None, ...] # expand num agents: num_groups, num_modes, 12, 2 -> 1, ...
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw.to(transformed_anchors.device)
        bbox_centers = bboxes.gravity_center.to(transformed_anchors.device)
        if with_rotation_transform: 
            angle = yaw - 3.1415953 # num_agents, 1
            rot_yaw = rot_2d(angle) # num_agents, 2, 2
            rot_yaw = rot_yaw[:, None, None,:, :] # num_agents, 1, 1, 2, 2
            transformed_anchors = rearrange(transformed_anchors, 'b g m t c -> b g m c t')  # 1, num_groups, num_modes, 12, 2 -> 1, num_groups, num_modes, 2, 12
            transformed_anchors = torch.matmul(rot_yaw, transformed_anchors)# -> num_agents, num_groups, num_modes, 12, 2
            transformed_anchors = rearrange(transformed_anchors, 'b g m c t -> b g m t c')
        if with_translation_transform:
            transformed_anchors = bbox_centers[:, None, None, None, :2] + transformed_anchors
        batched_anchors.append(transformed_anchors)
    return torch.stack(batched_anchors)


def trajectory_coordinate_transform(trajectory, bbox_results, with_translation_transform=True, with_rotation_transform=True):
    """
    Transform trajectory coordinates with respect to detected bounding boxes in the batch.
    Args:
        trajectory (torch.Tensor): predicted trajectory.
        bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
        with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
        with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.

    Returns:
        torch.Tensor: A tensor containing the transformed trajectory coordinates.
    """
    batch_size = len(bbox_results)
    batched_trajectories = []
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw.to(trajectory.device)
        bbox_centers = bboxes.gravity_center.to(trajectory.device)
        transformed_trajectory = trajectory[i,...]
        if with_rotation_transform:
            # we take negtive here, to reverse the trajectory back to ego centric coordinate
            angle = -(yaw - 3.1415953) 
            rot_yaw = rot_2d(angle)
            rot_yaw = rot_yaw[:,None, None,:, :] # A, 1, 1, 2, 2
            transformed_trajectory = rearrange(transformed_trajectory, 'a g p t c -> a g p c t') # A, G, P, 12 ,2 -> # A, G, P, 2, 12
            transformed_trajectory = torch.matmul(rot_yaw, transformed_trajectory)# -> A, G, P, 12, 2
            transformed_trajectory = rearrange(transformed_trajectory, 'a g p c t -> a g p t c')
        if with_translation_transform:
            transformed_trajectory = bbox_centers[:, None, None, None, :2] + transformed_trajectory
        batched_trajectories.append(transformed_trajectory)
    return torch.stack(batched_trajectories)