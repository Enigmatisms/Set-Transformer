"""
    Multi-head attention block
    @author hqy
    @date 2021.8.19
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch import nn

class Multihead(nn.Module):
    def __init__(self):
        super().__init__()


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
    def multiHeadQKVAtt(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_head = 8):
        pass

    # this might be based on Multi-head QKV Attension
    def forward(self, x):
        pass
