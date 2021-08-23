#-*-coding:utf-8-*-
"""
    Multi-head attention block
    @author hqy
    @date 2021.8.20
"""
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import nn
from math import sqrt

"""
    Input dim: Q, K, V (n, token_num, d_model)
    Output dim: Related to V, output is (n, token_num, dv_model)
    In Set transformer, token num is the length of sequence
"""
class Multihead(nn.Module):
    def __init__(self, head_num = 8, dk_model = 512, dv_model = 512):
        super().__init__()
        self.head_num = head_num
        self.dk = (dk_model // head_num)
        self.dv = (dv_model // head_num)
        dk_shape = (dk_model, self.dk)
        dv_shape = (dv_model, self.dv)
        # obviously, Wq, Wk, Wv can be nn.ModuleList([nn.Linear...])
        self.Wqs = nn.ParameterList([Parameter(torch.normal(0, 1, dk_shape), requires_grad = True) for _ in range(head_num)]) 
        self.Wks = nn.ParameterList([Parameter(torch.normal(0, 1, dk_shape), requires_grad = True) for _ in range(head_num)]) 
        self.Wvs = nn.ParameterList([Parameter(torch.normal(0, 1, dv_shape), requires_grad = True) for _ in range(head_num)]) 
        self.Wo = Parameter(torch.normal(0, 1, (dv_model, dv_model)))

    """
        Single head QKV attention
        Q K V might have the dim of batch, which could be (n_batch, n_token, embedding_dim)
        Therefore Q K V should be of (n_batch, n_token, embedding_dim)
    """
    def singleHeadQKVAtt(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        agreement = Q @ K.transpose(1, 2)
        dk = Q.shape[1]
        proba = F.softmax(agreement / sqrt(dk), dim = -1)
        return proba @ V

    # Multi-head QKV Attention function, which should map the Q K V to other dimensions
    def multiHeadQKVAtt(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        batch_size = Q.shape[0]
        Qs = [Q @ self.Wqs[i].repeat(batch_size, 1, 1) for i in range(self.head_num)]
        Ks = [K @ self.Wqs[i].repeat(batch_size, 1, 1) for i in range(self.head_num)]
        Vs = [V @ self.Wvs[i].repeat(batch_size, 1, 1) for i in range(self.head_num)]
        # each head outputs (n, token_num, token_num) @ (n, token_num, dv_model / head)
        heads = [self.singleHeadQKVAtt(Qs[i], Ks[i], Vs[i]) for i in range(self.head_num)]
        H = torch.cat(heads, dim = -1)
        # H is (n, token_num, dv_model)
        return H @ self.Wo.repeat(batch_size, 1, 1)      # (token_num, dv_model) * (dv_model, dv_model)

    # this might be based on Multi-head QKV Attension
    def forward(self, Q, K, V):
        return self.multiHeadQKVAtt(Q, K, V)
