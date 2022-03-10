from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model) # d_model=512, d_points = 32
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(# TODO :?好像有问题，没有联系局部关系，只是联系了xyz
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        # print(features.shape)
        dists = square_distance(xyz, xyz)
        # print('dists: ', dists.shape) # torch.Size([8, 1024, 1024])
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k 排序取前k（16）个
        # print('knn_idx: ', knn_idx.shape) # torch.Size([8, 1024, 16])
        knn_xyz = index_points(xyz, knn_idx)
        # print('knn_idx: ', knn_idx.shape) # torch.Size([8, 1024, 16, 3])
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        # torch.Size([8, 1024, 512])
        # torch.Size([8, 1024, 16, 512])
        # torch.Size([8, 1024, 16, 512])

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f torch.Size([8, 1024, 16, 512])
        
        # attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = self.fc_gamma(pos_enc) # torch.Size([8, 1024, 16, 512])
        attn = F.softmax(attn, dim=-2)  # b x n x k x f torch.Size([8, 1024, 16, 512])
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)# !haofangfa 矩阵元素对应相乘并求reduce sum
        res = self.fc2(res) + pre
        return res, attn
    