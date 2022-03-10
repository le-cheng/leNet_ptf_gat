import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pointnet_util import index_points, sample_and_group, PointNetSetAbstraction, square_distance
# from .transformer_copy import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction1(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model) # d_model=512, d_points = 32
        self.fc2 = nn.Linear(d_model, d_points)
        
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
    def forward(self, features):
        pre = features # 64
        x = self.fc1(features)
        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        kT = k.transpose(-1, -2)
        # attn = self.fc_gamma(torch.matmul(q,kT))
        attn = torch.matmul(q,kT)
        attn = F.softmax(attn/np.sqrt(k.size(-1)), dim=-1)  # b x n x k x f # TODO:哪个维度上比较好；测试-1，-2
        res = torch.matmul(attn, v)
        res = self.fc2(res) + pre
        return res


class PointNetSetAbstraction1(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction1, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            #in_channels,就是输入的四维张量[N, C, H, W]中的C了，即输入张量的channels数。
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.transformer = TransformerBlock(in_channel, 512, 16)
        self.pool = nn.MaxPool1d(self.nsample)
        self.pool2 = nn.MaxPool2d((1,16), stride=1)
        self.relu = nn.ReLU()
        # self.transformer0 = TransformerBlock(9, cfg.model.transformer_dim, nneighbor)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = self.transformer(new_points)


        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  self.relu(bn(conv(new_points)))
            # new_points =  F.relu(conv(new_points))

        # new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        
        # ppp = []
        # for point in new_points:
        #     ppp.append(self.pool(point.permute(0, 2, 1)).squeeze(-1))
        #     # x = self.pool(x.permute(0,1,3,2)).squeeze(-1)
        # # new_points = self.pool(new_points.permute(0,1,3,2)).squeeze(-1).permute(0, 2, 1)
        # new_points = torch.stack(ppp)
        # new_points = new_points.permute(0, 2, 1)

        new_points = self.pool2(new_points.permute(0, 3, 1, 2)).squeeze(-1)
        return new_xyz, new_points


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # print('le_net')
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        # npoints = 1024  nblocks = 4
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32), 
            nn.ReLU(),
            nn.Linear(32, 32) # point [16,1024,6] to [16,1024,32]
        )
        self.transformer0 = TransformerBlock(9, cfg.model.transformer_dim, nneighbor) #  cfg.model.transformer_dim= 512
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)

        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(
                npoints // 4 ** (i + 1), 
                nneighbor, 
                [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        self.nblocks = nblocks
        self.k = nneighbor
        self.fc_delta = nn.Sequential(# TODO :?好像有问题，没有联系局部关系，只是联系了xyz
            nn.Linear(9, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Linear(64, 32)
        )
        self.fc2 = nn.Linear(64, 32)
        self.pool = nn.MaxPool1d(16)
        self.pool2 = nn.MaxPool2d((1,16), stride=1)
        # input = Variable(torch.randn(2,2, 2, 3))
        self.linear1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.BatchNorm1d(out_planes),
            # nn.Linear(64, 32)
        )
       
    
    def forward(self, x):
        # print('-2: ',x.shape) # torch.Size([16, 1024, 6])
        xyz = x[..., :3]
        
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k 排序取前k（16）个
        # print('knn_idx: ', knn_idx.shape) # torch.Size([8, 1024, 16])
        # 分组
        knn_xyz_n = index_points(x, knn_idx) # b x n x k x 6
        xyz_pos = xyz[:, :, None] - knn_xyz_n[..., :3] # b x n x k x 3
        t = torch.cat((xyz_pos, knn_xyz_n), 3) 

        # print(t.shape)
        x = self.transformer0(t) # b x n x k x 9

        for i, layer in enumerate(self.fc_delta): 
            x = layer(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1) if i == 1 else layer(x)    # b x n x k x 32
       
        # x = self.fc_delta(x)  # b x n x k x 32 

        # x = torch.max(x, 2)[0]
        x = self.pool2(x.permute(0,1,3,2)).squeeze(-1) # torch.Size([16, 1024, 32])
        
        for i, layer in enumerate(self.linear1): 
            x = layer(x.permute(0, 2, 1)).permute(0, 2, 1) if i == 1 else layer(x)  # b x n x 32

        # ppp = []
        # for point in x:
        #     ppp.append(self.pool(point.permute(0, 2, 1)).squeeze(-1))
        #     # x = self.pool(x.permute(0,1,3,2)).squeeze(-1)
        # x = torch.stack(ppp)
        # print(x.shape)
        # raise
        # x = torch.einsum('bmnf->bmf', x) # b x n x 32

        # x = self.fc1(x) # b x n x 32
        # x = torch.cat((x, pos_enc), 2)  # 32 + 32
        points = self.transformer1(x) # TODO:
        # points = self.fc2(points)
        # print('-1: ',points.shape) # torch.Size([16, 1024, 32])

        for i in range(self.nblocks):
            xyz, points= self.transition_downs[i](xyz, points)
            # print(i,':',xyz.shape, points.shape)

            points = self.transformers[i](points)
            # print(i,':',points.shape)
            # 0 : torch.Size([16, 256, 3]) torch.Size([16, 256, 64])
            # 0 : torch.Size([16, 256, 64])
            # 1 : torch.Size([16, 64, 3]) torch.Size([16, 64, 128])
            # 1 : torch.Size([16, 64, 128])
            # 2 : torch.Size([16, 16, 3]) torch.Size([16, 16, 256])
            # 2 : torch.Size([16, 16, 256])
            # 3 : torch.Size([16, 4, 3]) torch.Size([16, 4, 512])
            # 3 : torch.Size([16, 4, 512])
        # raise NotImplementedError()
        return points

class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nblocks , n_c = cfg.model.nblocks, cfg.num_class
        # npoints, nneighbor, d_points = cfg.num_point,  cfg.model.nneighbor,  cfg.input_dim
        # self.nblocks = nblocks
        self.backbone = Backbone(cfg)
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256), # nblocks = 4
            # nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            # nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_c)
        )
        self.aap = nn.AdaptiveAvgPool1d(1)
        # self.flatten = nn.Flatten()

    def forward(self, x):
        # print(x.shape) # torch.Size([16, 1024, 6])
        points= self.backbone(x)
        # print(points.shape) # torch.Size([16, 4, 512])
        # res = self.fc2(points.mean(1))
        res = self.aap(points.permute(0, 2, 1)).squeeze(-1)
        res = self.fc2(res)
        res = F.log_softmax(res, -1)
        # print(res.shape) # torch.Size([16, 40])
        return res


    