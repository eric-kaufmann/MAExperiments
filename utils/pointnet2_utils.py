"""
Most of the code in this file is taken from the following repository:
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master
The PointNet++ Model was adapted for blood flow prediction. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def chunked_sort(group_idx, nsample, chunk_size=1024):
    """
    Sort the group_idx tensor in smaller chunks to reduce memory usage.
    
    Args:
        group_idx (torch.Tensor): The tensor to be sorted, shape [B, S, N].
        nsample (int): The number of samples to keep after sorting.
        chunk_size (int): The size of chunks to process at a time.
        
    Returns:
        torch.Tensor: The sorted tensor with shape [B, S, nsample].
    """
    B, S, N = group_idx.shape
    sorted_group_idx = torch.empty((B, S, nsample), dtype=group_idx.dtype, device=group_idx.device)
    
    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        sorted_chunk = group_idx[:, start:end, :].sort(dim=-1)[0][:, :, :nsample]
        sorted_group_idx[:, start:end, :] = sorted_chunk
    
    return sorted_group_idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    #group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_idx = chunked_sort(group_idx, nsample, chunk_size=2**10)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    PointNetSetAbstraction is a module for point cloud feature learning with set abstraction.
    Args:
        npoint (int): Number of points to sample.
        radius (float): Radius of the ball query.
        nsample (int): Number of samples in each ball query.
        in_channel (int): Number of input channels.
        mlp (list of int): List of output sizes for MLP layers.
        group_all (bool): Whether to group all points.
    Methods:
        forward(xyz, points):
            Forward pass of the module.
            Args:
                xyz (torch.Tensor): Input point cloud positions, shape (B, N, 3).
                points (torch.Tensor): Input point cloud features, shape (B, N, D).
            Returns:
                new_xyz (torch.Tensor): Sampled points positions, shape (B, npoint, 3).
                new_points (torch.Tensor): Sampled points features, shape (B, mlp[-1], npoint).
    """
    
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """
    PointNetSetAbstractionMsg is a class for PointNet++ set abstraction with multi-scale grouping (MSG).
    Args:
        npoint (int): Number of points to sample.
        radius_list (list of float): List of radii for ball query in each scale.
        nsample_list (list of int): List of number of samples in each ball query.
        in_channel (int): Number of input channels.
        mlp_list (list of list of int): List of MLP layers for each scale.
    Methods:
        forward(xyz, points):
            Forward pass of the network.
            Args:
                xyz (torch.Tensor): Input points position data, shape [B, C, N].
                points (torch.Tensor): Input points data, shape [B, D, N].
            Returns:
                new_xyz (torch.Tensor): Sampled points position data, shape [B, C, S].
                new_points_concat (torch.Tensor): Sampled points feature data, shape [B, D', S].
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
    
    
class PointNet2(nn.Module):
    """
    PointNet2 is a neural network model for point cloud feature learning and segmentation.
    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
    Attributes:
        sa1 (PointNetSetAbstraction): Set abstraction layer 1.
        sa2 (PointNetSetAbstraction): Set abstraction layer 2.
        sa3 (PointNetSetAbstraction): Set abstraction layer 3.
        sa4 (PointNetSetAbstraction): Set abstraction layer 4.
        fp4 (PointNetFeaturePropagation): Feature propagation layer 4.
        fp3 (PointNetFeaturePropagation): Feature propagation layer 3.
        fp2 (PointNetFeaturePropagation): Feature propagation layer 2.
        fp1 (PointNetFeaturePropagation): Feature propagation layer 1.
        conv1 (nn.Conv1d): Convolutional layer 1.
        bn1 (nn.BatchNorm1d): Batch normalization layer 1.
        drop1 (nn.Dropout): Dropout layer.
        conv2 (nn.Conv1d): Convolutional layer 2.
        lin1_1 (nn.Linear): Linear layer 1 for l4 points.
        lin1_2 (nn.Linear): Linear layer 2 for l4 points.
        lin1_3 (nn.Linear): Linear layer 3 for l4 points.
        lin2_1 (nn.Linear): Linear layer 1 for l3 points.
        lin2_2 (nn.Linear): Linear layer 2 for l3 points.
        lin2_3 (nn.Linear): Linear layer 3 for l3 points.
        final_lin1 (nn.Linear): Final linear layer 1.
        final_lin2 (nn.Linear): Final linear layer 2.
        final_lin3 (nn.Linear): Final linear layer 3.
        final_relu (nn.ReLU): ReLU activation function.
    Methods:
        forward(xyz):
            Forward pass of the network.
            Args:
                xyz (torch.Tensor): Input point cloud data of shape (B, N, C), where B is the batch size, 
                                    N is the number of points, and C is the number of channels.
            Returns:
                torch.Tensor: Output features of the network.
    """
    
    def __init__(self, c_in, c_out):
        super(PointNet2, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, c_in + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, c_out, 1)
        self.lin1_1 = nn.Linear(8192+48, 8192)
        self.lin1_2 = nn.Linear(8192, 8192)
        self.lin1_3 = nn.Linear(8192, 8192)
        self.lin2_1 = nn.Linear(16384+192, 16384)
        self.lin2_2 = nn.Linear(16384, 16384)
        self.lin2_3 = nn.Linear(16384, 16384)
        
        self.final_lin1 = nn.Linear(6144, 6144)
        self.final_lin2 = nn.Linear(6144, 6144)
        self.final_lin3 = nn.Linear(6144, 6144)
        self.final_relu = nn.ReLU()

    def forward(self, xyz):
        
        l0_points = xyz.permute(0, 2, 1)
        l0_xyz = xyz[:,:,:3].permute(0, 2, 1)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        l4_flat = torch.concatenate([l4_points.flatten(start_dim=1), l4_xyz.flatten(start_dim=1)], dim=1)
        lin1_1_out = F.relu(self.lin1_1(l4_flat))
        lin1_2_out = F.relu(self.lin1_2(lin1_1_out))
        lin1_3_out = F.relu(self.lin1_3(lin1_2_out))
        l4_points = lin1_3_out.reshape_as(l4_points)

        l3_flat = torch.concatenate([l3_points.flatten(start_dim=1), l3_xyz.flatten(start_dim=1)], dim=1)
        lin2_1_out = F.relu(self.lin2_1(l3_flat))
        lin2_2_out = F.relu(self.lin2_2(lin2_1_out))
        lin2_3_out = F.relu(self.lin2_3(lin2_2_out))
        l3_points = lin2_3_out.reshape_as(l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x_final_shape = x.shape
        
        x = self.final_lin3(x.flatten(start_dim=1))
        
        return x.reshape(x_final_shape)#, l4_points
    

