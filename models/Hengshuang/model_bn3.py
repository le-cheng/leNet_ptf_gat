import numpy as np
import torch
import torch.nn as nn
from pointnet_util import index_points, sample_and_group, PointNetSetAbstraction, square_distance


class TransitionDownBlock(nn.Module):
    def __init__(self, cfg, npoint, in_channel, out_channel, knn=True):
        super(TransitionDownBlock, self).__init__()
        self.npoint = npoint
        self.nneighbor = cfg.model.nneighbor
        self.transformer_dim = cfg.model.transformer_dim
        self.knn = knn

        self.transformer = LGTransformerBlock(in_channel, self.transformer_dim, lg= False)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, out_channel), 
            # nn.LayerNorm([self.nneighbor, out_channel]),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d((self.nneighbor, 1), stride=1)

    def forward(self, xyz, feature):
        """
        Input:
            xyz: input points position data, [B, N, C]
            festure: input points data with feature, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        new_xyz, new_feature = sample_and_group(xyz, feature, npoint=self.npoint, nsample=self.nneighbor, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_feature = self.transformer(new_feature)  # [B, npoint, nsample, C+D]

        for i, layer in enumerate(self.fc): 
            new_feature = layer(new_feature.permute(0, 3, 2, 1)).permute(0, 3, 2, 1) if i == 1 else layer(new_feature)
         # [B, npoint, nsample, out_channel]
        # new_feature = self.fc(new_feature)

        new_feature = self.pool2(new_feature).squeeze(-2)  # [B, npoint, out_channel]
        return new_xyz, new_feature  
    
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, bn_channel, dropout, lg= True, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.lg = lg
        self.dropout = nn.Dropout(dropout)
        if self.lg:
            self.bn = nn.BatchNorm1d(bn_channel)
        else:
            self.bn = nn.BatchNorm2d(bn_channel)

    def forward(self, X, Y):
        if self.lg:
            return self.bn((self.dropout(Y) + X).permute(0, 2, 1)).permute(0, 2, 1)
        else:
            return self.bn((self.dropout(Y) + X).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        # return self.dropout(Y) + X

class TransformerBlock(nn.Module):
    def __init__(self, in_channel, d_model) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channel, d_model) # d_model=512, d_points = 32
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.fc2 = nn.Linear(d_model, in_channel)

    # xyz: b x n x 3, features: b x n x f
    def forward(self, features):
        x = self.fc1(features)
        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        kT = k.transpose(-1, -2)
        # attn = self.fc_gamma(torch.matmul(q,kT))
        attn = torch.matmul(q,kT)
        attn = self.softmax(attn/np.sqrt(k.size(-1)))  # b x n x k x f # TODO:哪个维度上比较好；测试-1，-2
        res = torch.matmul(attn, v)
        # res = self.fc2(res)+features
        res = self.fc2(res)
        return res

class LGTransformerBlock(nn.Module):
    def __init__(self, in_channel, d_model, lg= True, ndropout=0.1) -> None:
        super().__init__()
        self.transformer = TransformerBlock(in_channel, d_model)
        self.addnorm0 = AddNorm(bn_channel=in_channel, dropout=ndropout, lg= lg)
        self.ffn = PositionWiseFFN(ffn_num_input=in_channel, ffn_num_hiddens=in_channel, ffn_num_outputs=in_channel)
        self.addnorm1 = AddNorm(bn_channel=in_channel, dropout=ndropout, lg= lg)

    def forward(self, features):
        res = self.addnorm0(features, self.transformer(features))
        # return self.addnorm1(res, self.ffn(res))
        return res


# class LocalTransformerBlock(nn.Module):
#     def __init__(self, in_channel, d_model, ndropout=0.1) -> None:
#         super().__init__()
#         self.transformer = TransformerBlock(in_channel, d_model)
#         self.addnorm0 = AddNorm(bn_channel=in_channel, dropout=ndropout, lg= False)
#         self.ffn = PositionWiseFFN(ffn_num_input=in_channel, ffn_num_hiddens=in_channel, ffn_num_outputs=in_channel)
#         self.addnorm1 = AddNorm(bn_channel=in_channel, dropout=ndropout, lg= False)

#     # xyz: b x n x 3, features: b x n x f
#     def forward(self, features):
#         res = self.transformer(features)
#         res = self.addnorm0(features, res)
#         res = self.addnorm1(res, self.ffn(res))
#         return res

# class GlobalTransformerBlock(nn.Module):
#     def __init__(self, in_channel, d_model, ndropout=0.1) -> None:
#         super().__init__()
#         self.transformer = TransformerBlock(in_channel, d_model)
#         self.addnorm0 = AddNorm(bn_channel=in_channel, dropout=ndropout, lg= True)
#         self.ffn = PositionWiseFFN(ffn_num_input=in_channel, ffn_num_hiddens=in_channel, ffn_num_outputs=in_channel)
#         self.addnorm1 = AddNorm(bn_channel=in_channel, dropout=ndropout, lg= True)

#     # xyz: b x n x 3, features: b x n x f
#     def forward(self, features):
#         res = self.transformer(features)
#         res = self.addnorm0(features, res)
#         res = self.addnorm1(res, self.ffn(res))
#         return res

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, self.nblocks = cfg.num_point, cfg.model.nblocks # npoints = 1024  nblocks = 4
        self.tdb = TransitionDownBlock(cfg, npoints, 6, 32)
        self.transformer0 = LGTransformerBlock(32, cfg.model.transformer_dim, lg = True) #  cfg.model.transformer_dim= 512

        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(self.nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDownBlock(
                cfg,
                npoints // 4 ** (i + 1), 
                (channel // 2 + 3), 
                channel
                ))
            self.transformers.append(LGTransformerBlock(
                channel, 
                # npoints // 4 ** (i + 1),
                cfg.model.transformer_dim, 
                lg = True
                ))

    def forward(self, x):
        # torch.Size([16, 1024, 6])
        xyz = x[..., :3]# torch.Size([16, 1024, 3])
        xyz, points = self.tdb(xyz, xyz)
        points = self.transformer0(points) # torch.Size([16, 1024, 32])
        for i in range(self.nblocks):
            xyz, points= self.transition_downs[i](xyz, points)
            for j in range(i+2):
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
        self.backbone = Backbone(cfg)
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256), # nblocks = 4
            # nn.BatchNorm1d(256), 
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            # nn.BatchNorm1d(64), 
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_class)
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # torch.Size([16, 1024, 6])
        points= self.backbone(x)  # torch.Size([16, 4, 512])
        res = self.aap(points.permute(0, 2, 1)).squeeze(-1)  # [16,512]
        res = self.fc2(res)  # [16,40]
        res = self.log_softmax(res)  # torch.Size([16, 40])
        return res


