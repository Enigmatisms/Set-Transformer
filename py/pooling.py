#-*-coding:utf-8-*-
"""
    Pooling module: PMA (Pooling by Multi-head Attention)
    @author hqy
    @date 2021.8.21
"""

import torch
from torch import nn
from att_blocks import MAB, SAB
from torch.nn.parameter import Parameter

"""
    Since PMA is used in the decoders, we assume that a FFN always presents before the decoder
    The output of PMA = MAB(S, Z), Z is from the Encoder
    S has shape: (n, k_seeds, n_embedding_dim), which is similar to inducing points
"""
class PMA(nn.Module):
    def __init__(self, batch_size, k_seeds = 1, head_num = 8, d_model = 512, use_layer_norm = True):
        super().__init__()
        self.seeds = Parameter(torch.normal(0, 1, (batch_size, k_seeds, d_model)))
        self.mab = MAB(batch_size, head_num, d_model, d_model, use_layer_norm)
        self.lin = nn.Linear(d_model, d_model)
        self.sab = None
        if k_seeds > 1:
            self.sab = SAB(batch_size, head_num, d_model, use_layer_norm)
        """
            For the output of SAB(PMA(S, Z)), the followings hold:
            - PMA output has shape: (n, k, d_model)
            - SAB is MAB(X, X), therefore outputs (n, token_num, d_model), which is (n, k, d)
        """

    def forward(self, Z):
        H = self.mab(self.seeds, Z)
        if self.sab is None:
            return self.lin(H)
        return self.lin(self.sab(H))        # transform the last dim
