import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    [dis,idx] = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return idx
    # idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    # return idx

def knn_d(x_n):
    mid = x_n[:,:,1:,:].sum(-2)
    dis = torch.sum((x_n[:,:,0,:]-mid)**2, dim=-1)
    dis_norm = dis*(dis.max(-1)[0].view(x_n.shape[0],1)**-1)
    dis_norm = F.softmax(dis_norm,-1)
    return dis_norm

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    dis_norm = knn_d(feature)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature ,dis_norm

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k=16) -> None:
        super().__init__()
        # self.fc1 = nn.Linear(d_points, d_model) # d_model=512, d_points = 32  换nn.Conv1d
        self.w_qs = nn.Linear(d_points, d_model, bias=False)
        self.w_ks = nn.Linear(d_points, d_model, bias=False)
        self.w_vs = nn.Linear(d_points, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.fc2 = nn.Linear(d_model, d_points)
        self.bn1 = nn.BatchNorm1d(d_points)
        self.relu = nn.ReLU()
    # xyz: b x n x 3, features: b x n x f
    def forward(self, features):
        pre = features # 64
        x = features
        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        kT = k.transpose(-1, -2)
        # attn = self.fc_gamma(torch.matmul(q,kT))
        attn = torch.matmul(q,kT)
        attn = self.softmax(attn/np.sqrt(k.size(-1)))  # b x n x k x f # TODO:哪个维度上比较好；测试-1，-2
        res = torch.matmul(attn, v)

        res = F.leaky_relu(self.fc2(res).permute(0, 2, 1).permute(0, 2, 1), negative_slope=0.2)
        # res = self.fc2(res) + pre
        res = res + pre
        return res


class PointTransformerCls(nn.Module):
    def __init__(self, cfg, output_channels=40):
        super(PointTransformerCls, self).__init__()
        # self.args = args
        self.k = 20
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        # self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.transformer0 = TransformerBlock(64, 8)
        self.transformer1 = TransformerBlock(64, 8)
        self.transformer2 = TransformerBlock(128, 8)
        self.transformer3 = TransformerBlock(256, 8)
        self.transformer4 = TransformerBlock(1024, 8)

        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x0 = x.permute(0,2,1)
        batch_size = x0.size(0)
        # print('x ',x.shape)
        # torch.Size([16, 3, 1024])
        x, dis_norm = get_graph_feature(x0, k=self.k)  # torch.Size([16, 6, 1024, 20])
        # print('x ',x.shape)
        x = self.conv1(x)  # torch.Size([16, 64, 1024, 20])
        # print('x ',x.shape)

        x1 = x.max(dim=-1, keepdim=False)[0]  # torch.Size([16, 64, 1024])
        Centrality_Encoding = dis_norm.view(batch_size,1,-1).repeat(1,x1.shape[1],1) # 数值稳定性
        x1 = x1 * Centrality_Encoding #节点重要性编码 
        # 缺点，Transformer是全局视野，不能有效的使用图的结构信息。
        x1 = self.transformer0(x1.permute(0, 2, 1)).permute(0, 2, 1)


        x, dis_norm= get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]  # torch.Size([16, 64, 1024])
        # print('x2 ',x2.shape)
        Centrality_Encoding = dis_norm.view(batch_size,1,-1).repeat(1,x2.shape[1],1) # 数值稳定性
        x2 = x2 * Centrality_Encoding #节点重要性编码 
        x2 = self.transformer1(x2.permute(0, 2, 1)).permute(0, 2, 1)

        x, dis_norm= get_graph_feature(x2, k=self.k)  
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]  # torch.Size([16, 128, 1024])
        # print('x3 ',x3.shape)
        Centrality_Encoding = dis_norm.view(batch_size,1,-1).repeat(1,x3.shape[1],1) # 数值稳定性
        x3 = x3 * Centrality_Encoding #节点重要性编码 
        x3 = self.transformer2(x3.permute(0, 2, 1)).permute(0, 2, 1)

        x, dis_norm= get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]  # torch.Size([16, 256, 1024])
        # print('x4 ',x4.shape)
        Centrality_Encoding = dis_norm.view(batch_size,1,-1).repeat(1,x4.shape[1],1) # 数值稳定性
        x4 = x4 * Centrality_Encoding #节点重要性编码 
        x4 = self.transformer3(x4.permute(0, 2, 1)).permute(0, 2, 1)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x