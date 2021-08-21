"""
    Multi-head attention block
    @author hqy
    @date 2021.8.19
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch.nn.parameter import Parameter
from torch import nn

class Multihead(nn.Module):
    def __init__(self, batch_size, head_num = 8, dk_model = 512, dv_model = 512):
        super().__init__()
        self.dk = (dk_model // head_num)
        self.dv = (dv_model // head_num)
        self.Wqs = nn.ParameterList([Parameter(torch.normal(batch_size, dk_model, self.dk)) for _ in range(head_num)]) 
        self.Wvs = nn.ParameterList([Parameter(torch.normal(batch_size, dv_model, self.dv)) for _ in range(head_num)]) 
        self.Wo = Parameter(torch.normal(batch_size, dv_model, dv_model))
        self.batch_size = batch_size

    """
        Single head QKV attention
        Q K V might have the dim of batch, which could be (n_batch, n_token, embedding_dim)
        Therefore Q K V should be of (n_batch, n_token, embedding_dim)
    """
    def singleHeadQKVAtt(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        agreement = Q @ K.T
        dk = Q.shape[1]
        proba = F.softmax(agreement / torch.sqrt(dk), dim = -1)
        return proba @ V

    # Multi-head QKV Attention function, which should map the Q K V to other dimensions
    def multiHeadQKVAtt(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        Qs = [Q @ self.Wqs[i] for i in range(self.head_num)]
        Ks = [K @ self.Wqs[i] for i in range(self.head_num)]
        Vs = [V @ self.Wvs[i] for i in range(self.head_num)]
        heads = [self.singleHeadQKVAtt(Qs[i], Ks[i], Vs[i]) for i in range(self.head_num)]
        H = torch.cat(heads, dim = -1)
        return H @ self.Wo

    # this might be based on Multi-head QKV Attension
    def forward(self, Q, K, V):
        return self.multiHeadQKVAtt(Q, K, V)
