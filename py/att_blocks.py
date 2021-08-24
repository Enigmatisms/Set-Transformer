"""
    Mutable input output attention block
    @author hqy
    @date 2021.8.22
"""

import torch
from torch import nn
from multihead import Multihead
from torch.nn.parameter import Parameter

# MAB output shape (n, x_token, dim_v)
class MAB(nn.Module):
    def __init__(self, head_num, dim_q, dim_k, dim_v, use_layer_norm = True, split = False):
        super().__init__()
        self.lin_dq = nn.Linear(dim_q, dim_v)
        self.lin_dk = nn.Linear(dim_k, dim_v)
        # K, V should be different, therefore adapts different transformation
        # In MAB, MAB(Q, K), K = V, therefore, input should be transformed
        self.lin_dv = nn.Linear(dim_k, dim_v)
        # for Multihead Attention Block, Q & K must have the same dim
        self.att = Multihead(head_num, dim_v, dim_v, split)
        self.ln0 = self.ln1 = None
        if use_layer_norm:
            self.ln0 = nn.LayerNorm(dim_v)
            self.ln1 = nn.LayerNorm(dim_v)
        self.ff = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.ReLU(True)
        )

    def forward(self, X, Y):
        if X.dim() < 3:
            X = X.unsqueeze(dim = -1)
        if Y.dim() < 3:
            Y = Y.unsqueeze(dim = -1)
        Q = self.lin_dq(X)
        K = self.lin_dk(Y)
        V = self.lin_dv(Y)
        H = Q + self.att(Q, K, V)
        if not self.ln0 is None:
            H = self.ln0(H)
        H = self.ff(H) + H
        if not self.ln1 is None:
            return self.ln1(H)
        return H

# SAB outputs (n_batch, x_token, dim_out)
class SAB(nn.Module):
    def __init__(self, head_num, dim_in, dim_out, use_layer_norm = True, split = False):
        super().__init__()
        self.mab = MAB(head_num, dim_in, dim_in, dim_out, use_layer_norm, split)
    
    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, head_num, m, dim_in, dim_out, use_layer_norm = True, split = False):
        super().__init__()
        self.inducing_point = Parameter(torch.normal(0, 1, (m, dim_in)))
        self.mab1 = MAB(head_num, dim_in, dim_in, dim_out, use_layer_norm, split)
        self.mab2 = MAB(head_num, dim_out, dim_out, dim_out, use_layer_norm, split)

    def forward(self, X):
        H = self.mab1(self.inducing_point.repeat(X.shape[0], 1, 1), X)
        return self.mab2(X, H)
