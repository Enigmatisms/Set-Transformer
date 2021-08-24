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
    def __init__(self, head_num = 8, dk_model = 512, dv_model = 512, split = False):
        super().__init__()
        self.head_num = head_num
        self.dk = (dk_model // head_num)
        self.dv = (dv_model // head_num)
        dk_shape = (dk_model, self.dk)
        dv_shape = (dv_model, self.dv)
        # obviously, Wq, Wk, Wv can be nn.ModuleList([nn.Linear...])
        self.Wqs, self.Wks, self.Wvs = None, None, None
        if split == False:
            self.Wqs = nn.ParameterList([Parameter(torch.normal(0, 1, dk_shape), requires_grad = True) for _ in range(head_num)]) 
            self.Wks = nn.ParameterList([Parameter(torch.normal(0, 1, dk_shape), requires_grad = True) for _ in range(head_num)]) 
            self.Wvs = nn.ParameterList([Parameter(torch.normal(0, 1, dv_shape), requires_grad = True) for _ in range(head_num)]) 
        self.Wo = nn.Linear(dv_model, dv_model)
        nn.init.normal_(self.Wo.weight)
        nn.init.normal_(self.Wo.bias)
        print("Multihead split: ", split)
        self.split = split

    """
        Single head QKV attention
        Q K V might have the dim of batch, which could be (n_batch, n_token, embedding_dim)
        Therefore Q K V should be of (n_batch, n_token, embedding_dim)
    """
    def singleHeadQKVAtt(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        agreement = Q @ K.transpose(1, 2)
        dk = Q.shape[2]
        proba = F.softmax(agreement / sqrt(dk), dim = -1)
        return proba @ V

    # Multi-head QKV Attention function, which should map the Q K V to other dimensions
    # What I found is that, the author of Set Transformer didn't do the re-map, he performed direct splitting!
    def multiHeadQKVAtt(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        dim_v = Q.shape[2]
        batch_size = Q.shape[0]
        if self.split == True:
            Qs = Q.split(self.dk, dim = -1)
            Ks = K.split(self.dk, dim = -1)
            Vs = V.split(self.dv, dim = -1)
            heads = [self.singleHeadQKVAtt(Qs[i], Ks[i], Vs[i]) for i in range(self.head_num)]
            H = torch.cat(heads, dim = -1)
            return self.Wo(H)
        else:
            Qs = [Q @ self.Wqs[i].repeat(batch_size, 1, 1) for i in range(self.head_num)]
            Ks = [K @ self.Wqs[i].repeat(batch_size, 1, 1) for i in range(self.head_num)]
            Vs = [V @ self.Wvs[i].repeat(batch_size, 1, 1) for i in range(self.head_num)]
            # each head outputs (n, token_num, token_num) @ (n, token_num, dv_model / head)
            heads = [self.singleHeadQKVAtt(Qs[i], Ks[i], Vs[i]) for i in range(self.head_num)]
            H = torch.cat(heads, dim = -1)
            # H is (n, token_num, dv_model)
            return self.Wo(H)      # (token_num, dv_model) * (dv_model, dv_model)
        # Q_ = torch.cat(Q.split(self.dv, 2), 0)
        # K_ = torch.cat(K.split(self.dv, 2), 0)
        # V_ = torch.cat(V.split(self.dv, 2), 0)

        # A = torch.softmax(Q_.bmm(K_.transpose(1,2))/ sqrt(dim_v), 2)
        # O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        # return O

    # this might be based on Multi-head QKV Attension
    def forward(self, Q, K, V):
        return self.multiHeadQKVAtt(Q, K, V)
