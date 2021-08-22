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
    def __init__(self, batch_size, head_num, dim_q, dim_k, dim_v, use_layer_norm = True):
        super().__init__()
        self.lin_dq = nn.Linear(dim_q, dim_v)
        self.lin_dk = nn.Linear(dim_k, dim_v)
        # K, V should be different, therefore adapts different transformation
        # In MAB, MAB(Q, K), K = V, therefore, input should be transformed
        self.lin_dv = nn.Linear(dim_k, dim_v)
        # for Multihead Attention Block, Q & K must have the same dim
        self.att = Multihead(batch_size, head_num, dim_v, dim_v)
        self.ln0 = self.ln1 = None
        if use_layer_norm:
            self.ln0 = nn.LayerNorm(dim_v)
            self.ln1 = nn.LayerNorm(dim_v)
        self.ff = nn.Linear(dim_v, dim_v)

    def forward(self, X, Y):
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
    def __init__(self, batch_size, head_num, dim_in, dim_out, use_layer_norm = True):
        super().__init__()
        self.mab = MAB(batch_size, head_num, dim_in, dim_in, dim_out, use_layer_norm)
    
    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, batch_size, head_num, m, dim_in, dim_out, use_layer_norm = True):
        super().__init__()
        self.inducing_point = Parameter(torch.normal(0, 1, (batch_size, m, dim_in)))
        self.mab1 = MAB(batch_size, head_num, dim_in, dim_in, dim_out, use_layer_norm)
        self.mab2 = MAB(batch_size, head_num, dim_out, dim_out, dim_out, use_layer_norm)

    def forward(self, X):
        H = self.mab1(self.inducing_point, X)
        return self.mab2(X, H)
