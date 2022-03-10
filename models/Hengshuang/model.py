import torch
import torch.nn as nn
from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from .transformer import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32), 
            nn.ReLU(),
            nn.Linear(32, 32) # point [16,1024,6] to [16,1024,32]
        )
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        self.nblocks = nblocks
    
    def forward(self, x):
        # print('-2: ',x.shape) # torch.Size([16, 1024, 6])
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]
        # print('-1: ',points.shape) # torch.Size([16, 1024, 32])

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            # print(i,':',xyz.shape, points.shape)
            points = self.transformers[i](xyz, points)[0]
            # print(i,':',points.shape)
            xyz_and_feats.append((xyz, points))
            # 0 : torch.Size([16, 256, 3]) torch.Size([16, 256, 64])
            # 0 : torch.Size([16, 256, 64])
            # 1 : torch.Size([16, 64, 3]) torch.Size([16, 64, 128])
            # 1 : torch.Size([16, 64, 128])
            # 2 : torch.Size([16, 16, 3]) torch.Size([16, 16, 256])
            # 2 : torch.Size([16, 16, 256])
            # 3 : torch.Size([16, 4, 3]) torch.Size([16, 4, 512])
            # 3 : torch.Size([16, 4, 512])

        # raise NotImplementedError()
        return points, xyz_and_feats


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        nblocks , n_c = cfg.model.nblocks, cfg.num_class
        # npoints, nneighbor, d_points = cfg.num_point,  cfg.model.nneighbor,  cfg.input_dim
        # self.nblocks = nblocks
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256), # nblocks = 4
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
    
    def forward(self, x):
        # print(x.shape) # torch.Size([16, 1024, 6])
        points, _ = self.backbone(x)
        # print(points.shape) # torch.Size([16, 4, 512])
        res = self.fc2(points.mean(1))
        # print(res.shape) # torch.Size([16, 10])
        return res


class PointTransformerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, cfg.model.transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))

        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
    
    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]
            
        return self.fc3(points)


    