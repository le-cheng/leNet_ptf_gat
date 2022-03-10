import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pointnet_util import index_points, sample_and_group, PointNetSetAbstraction, square_distance


class TransitionDownBlock(nn.Module):
    def __init__(self, cfg, npoint, in_channel, out_channel, knn=True):
        super(TransitionDownBlock, self).__init__()
        self.npoint = npoint
        self.nneighbor = cfg.model.nneighbor
        self.transformer_dim = cfg.model.transformer_dim
        self.knn = knn

        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, in_channel), 
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.transformer = TransformerBlock(in_channel, self.transformer_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, out_channel), 
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(out_channel*2, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        # self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #     #in_channels,就是输入的四维张量[N, C, H, W]中的C了，即输入张量的channels数。
        # self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        # last_channel = out_channel
        self.pool1 = nn.AvgPool2d((self.nneighbor, 1), stride=1)
        self.pool2 = nn.MaxPool2d((self.nneighbor, 1), stride=1)

    def forward(self, xyz, festure):
        """
        Input:
            xyz: input points position data, [B, N, C]
            festure: input points data with feature, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        new_xyz, new_festure = sample_and_group(xyz, festure, npoint=self.npoint, nsample=self.nneighbor, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # # new_points: sampled points data, [B, npoint, nsample, C+D]
        # for i, layer in enumerate(self.fc1): 
        #     new_festure = layer(new_festure.permute(0, 3, 2, 1)).permute(0, 3, 2, 1) if i == 1 else layer(new_festure)
        new_festure = self.mlp1(new_festure.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        new_festure = self.transformer(new_festure)  # [B, npoint, nsample, C+D]

        # for i, layer in enumerate(self.fc2): 
        #     new_festure = layer(new_festure.permute(0, 3, 2, 1)).permute(0, 3, 2, 1) if i == 1 else layer(new_festure)
        #  # [B, npoint, nsample, out_channel]

        new_festure = self.mlp2(new_festure.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        # for i, conv in enumerate(self.mlp_convs):
        #     bn = self.mlp_bns[i]
        #     new_points =  self.relu(bn(conv(new_points)))

        # new_festure_avg = self.pool1(new_festure).squeeze(-2)  # [B, npoint, out_channel]
        new_festure = self.pool2(new_festure).squeeze(-2)  # [B, npoint, out_channel]
        # new_festure = torch.cat((new_festure_avg, new_festure_max), dim=-1)

        # new_festure = self.mlp3(new_festure.permute(0, 2, 1)).permute(0, 2, 1)

        return new_xyz, new_festure  

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k=16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model) # d_model=512, d_points = 32
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.fc2 = nn.Linear(d_model, d_points)
    # xyz: b x n x 3, features: b x n x f
    def forward(self, features):
        pre = features # 64
        x = self.fc1(features)
        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        kT = k.transpose(-1, -2)
        # attn = self.fc_gamma(torch.matmul(q,kT))
        attn = torch.matmul(q,kT)
        attn = self.softmax(attn/np.sqrt(k.size(-1)))  # b x n x k x f # TODO:哪个维度上比较好；测试-1，-2
        res = torch.matmul(attn, v)
        res = self.fc2(res) + pre
        return res

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nneighbor = cfg.model.nneighbor
        npoints, self.nblocks= cfg.num_point, cfg.model.nblocks
        # npoints = 1024  nblocks = 4

        self.tdb = TransitionDownBlock(cfg, npoints, 6, 32)
        self.transformer0 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor) #  cfg.model.transformer_dim= 512
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(self.nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDownBlock(
                cfg,
                npoints // 4 ** (i + 1), 
                (channel // 2 + 3), 
                channel))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        # self.nblocks = nblocks
        
    def forward(self, x):
        # print('-2: ',x.shape) # torch.Size([16, 1024, 6])
        xyz = x[..., :3]

        xyz, points = self.tdb(xyz, x)
        points = self.transformer0(points) 
        for i in range(self.nblocks):
            xyz, points= self.transition_downs[i](xyz, points)
            # print(i,':',xyz.shape, points.shape)
            # for j in range(i+2):
            points = self.transformers[i](points)
           # 0 : torch.Size([16, 256, 64])
            # 1 : torch.Size([16, 64, 128])
            # 2 : torch.Size([16, 16, 256])
            # 3 : torch.Size([16, 4, 512])
            # raise NotImplementedError()
        return points

class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nblocks , n_class = cfg.model.nblocks, cfg.num_class
        # npoints, nneighbor, d_points = cfg.num_point,  cfg.model.nneighbor,  cfg.input_dim
        # self.nblocks = nblocks
        self.backbone = Backbone(cfg)
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256), # nblocks = 4
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            # nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_class)
        )
        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Linear(1024 , 512, bias=False)
        self.conv2 = nn.Linear(512, n_class)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # torch.Size([16, 1024, 6])
        points= self.backbone(x)  # torch.Size([16, 4, 512])

        # x = self.conv0(points.permute(0,2,1))
        # x_max = F.adaptive_max_pool1d(x, 1)
        # x_avg = F.adaptive_avg_pool1d(x, 1)
        
        # x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        # x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        # x = self.dp1(x)
        # res = self.conv2(x)

        res = self.aap(points.permute(0, 2, 1)).squeeze(-1)  # [16,512]
        res = self.fc2(res)  # [16,40]
        # res = self.log_softmax(res)  # torch.Size([16, 40])
        return res


