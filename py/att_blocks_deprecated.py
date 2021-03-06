#-*-coding:utf-8-*-
"""
    Deprecated Set Transformer Attention Blocks
    MAB SAB ISAB Modules
    @author hqy
    @date 2021.8.21
"""

import torch
from torch import nn
from multihead import Multihead
from torch.nn.parameter import Parameter

"""
    what is rFF in the paper? I just treat rFF as a single-layer perceptrons
    Multi-head Attension Block, X(b, n1, k) Y(b, n2, k), which outputs (b, n1, k)
    batch_size / X token num (which is usually eual to Y) / Y(X, the same) dimensionality
    X, Y input might be in the smae dimension, yet we might not want it to be
"""
class MAB(nn.Module):
    def __init__(self, batch_size, head_num = 8, dk_model = 512, dv_model = 512, use_layer_norm = True):
        super().__init__()
        self.att = Multihead(batch_size, head_num, dk_model, dv_model, use_layer_norm)
        self.ff = nn.Linear(dv_model, dv_model)
        self.layer_norm = None
        if use_layer_norm == True:
            self.layer_norm = nn.LayerNorm(dv_model)
        self.remap = nn.Linear(dk_model, dv_model)

    def forward(self, X, Y):
        H = self.att(X, Y, Y) + X
        if not self.layer_norm is None:
            H = self.layer_norm(H)
        H = H + self.ff(H)
        if not self.layer_norm is None:
            H = self.layer_norm(H)
        return H

# output is (n, token_x, embedding_x)
class SAB(nn.Module):
    def __init__(self, batch_size, head_num = 8, d_model = 512, use_layer_norm = True):
        super().__init__()
        self.mab = MAB(batch_size, head_num, d_model, use_layer_norm)
    
    def forward(self, X):
        return self.mab(X, X)

# Output is (n, m_induce, d_model)
class ISAB(nn.Module):
    def __init__(self, batch_size, head_num = 8, d_model = 512, m_induce = 16, use_layer_norm = True):
        super().__init__()
        # This is correct, the normal input should have (n_batch, n_token, n_embedding_dim)
        shape = (batch_size, m_induce, d_model)
        self.inducing_point = Parameter(torch.normal(0, 1, shape), requires_grad = True)
        self.mab1 = MAB(batch_size, head_num, d_model, use_layer_norm)
        self.mab2 = MAB(batch_size, head_num, d_model, use_layer_norm)

    def forward(self, X):
        # H is (n, m, k), which is fed to mab2: (n, d, k) * (n, k, m) * (n, m, k) -> (n, d, k)
        H = self.mab1(self.inducing_point, X)
        return self.mab2(X, H)
