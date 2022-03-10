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
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


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
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

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
        attn = self.softmax(attn/np.sqrt(k.size(-1)))  # b x n x k x f 
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

        self.transformer0 = EncoderLayer(n_points = 1024, hidden_size = 64, ffn_size = 8, 
            dropout_rate = 0.1, 
            attention_dropout_rate = 0.1, 
            num_heads = 8)
        self.transformer1 = EncoderLayer(n_points = 1024, hidden_size = 64, ffn_size = 8, 
            dropout_rate = 0.1, 
            attention_dropout_rate = 0.1, 
            num_heads = 4)
        self.transformer2 = EncoderLayer(n_points = 1024, hidden_size = 64, ffn_size = 8, 
            dropout_rate = 0.1, 
            attention_dropout_rate = 0.1, 
            num_heads = 4)
        self.transformer3 = EncoderLayer(n_points = 1024, hidden_size = 128, ffn_size = 8, 
            dropout_rate = 0.1, 
            attention_dropout_rate = 0.1, 
            num_heads = 4)
        self.transformer4 = EncoderLayer(n_points = 1024, hidden_size = 256, ffn_size = 8, 
            dropout_rate = 0.1, 
            attention_dropout_rate = 0.1, 
            num_heads = 4)
        # self.transformer0 = TransformerBlock(64, 8)
        # self.transformer1 = TransformerBlock(64, 8)
        # self.transformer2 = TransformerBlock(64, 8)
        # self.transformer3 = TransformerBlock(128, 8)
        # self.transformer4 = TransformerBlock(256, 8)

        # self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
                                # nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
                                # nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
                                # nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
                                # nn.ReLU())
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

        x = get_graph_feature(x0, k=self.k)  # torch.Size([16, 6, 1024, 20])
        x = self.conv1(x)  # torch.Size([16, 64, 1024, 20])
        x1 = x.max(dim=-1, keepdim=False)[0]  # torch.Size([16, 64, 1024])
        x1 = self.transformer1(x1.permute(0, 2, 1)).permute(0, 2, 1)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]  # torch.Size([16, 64, 1024])
        x2 = self.transformer2(x2.permute(0, 2, 1)).permute(0, 2, 1)

        x = get_graph_feature(x2, k=self.k)  
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]  # torch.Size([16, 128, 1024])
        x3 = self.transformer3(x3.permute(0, 2, 1)).permute(0, 2, 1)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]  # torch.Size([16, 256, 1024])
        x4 = self.transformer4(x4.permute(0, 2, 1)).permute(0, 2, 1)

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



class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        # self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.layer1 = nn.Conv1d(hidden_size, ffn_size , kernel_size=1)
        # self.gelu = nn.GELU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.layer2 = nn.Conv1d(ffn_size, hidden_size , kernel_size=1)
        # self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        x = self.LeakyReLU(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_size = att_size = 4
        # hidden_size // num_heads
        self.scale = att_size ** -0.5
        
        self.linear_q = nn.Conv1d(hidden_size, num_heads * att_size , kernel_size=1, bias=False)
        self.linear_k = nn.Conv1d(hidden_size, num_heads * att_size , kernel_size=1, bias=False)
        self.linear_v = nn.Conv1d(hidden_size, num_heads * att_size , kernel_size=1, bias=False)

        # self.linear_q = nn.Linear(hidden_size, num_heads * att_size, bias=False)
        # self.linear_k = nn.Linear(hidden_size, num_heads * att_size, bias=False)
        # self.linear_v = nn.Linear(hidden_size, num_heads * att_size, bias=False)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Sequential(nn.Conv1d(num_heads * att_size , hidden_size, kernel_size=1),
                                          nn.BatchNorm1d(hidden_size),
                                          nn.LeakyReLU(negative_slope=0.2))
        
        # self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, d_k, -1, self.num_heads)
        k = self.linear_k(k).view(batch_size, d_k, -1, self.num_heads)
        v = self.linear_v(v).view(batch_size, d_v, -1, self.num_heads)

        # q = q.transpose(1, 2)  
        q = q.permute(0, 3, 2, 1)                # [b, h, q_len, d_k]
        # v = v.transpose(1, 2)     
        v = v.permute(0, 3, 2, 1)                # [b, h, v_len, d_v]
        k = k.permute(0, 3, 1, 2)                # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        # q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x = x * self.scale
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=-1)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(3, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, self.num_heads * d_v, -1)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, n_points, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()
        assert hidden_size >= 64
        assert hidden_size % 8 == 0

        self.self_attention_norm = nn.LayerNorm([hidden_size, n_points])
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm([hidden_size, n_points])
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        x = x.permute(0, 2, 1).contiguous()
        # y = self.self_attention_norm(x)
        y = x
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        # y = self.ffn_norm(x)
        # y = x
        # y = self.ffn(y)
        # y = self.ffn_dropout(y)
        # x = x + y
        x = x.permute(0, 2, 1).contiguous()
        return x
