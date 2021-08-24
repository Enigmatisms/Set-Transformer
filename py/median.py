"""
    A set learning problem, hereinafter we wish the model is able to learn
    The median number of the set. In the paper of set transformer, max num learning 
    is experimented.
    @author sentinel
    @date 2021.8.22
"""
import torch
from torch import nn
from att_blocks import MAB, SAB, ISAB
from pooling import PMA

class Median(nn.Module):
    def __init__(self, head_num = 4, use_layer_norm = True, split = False):
        super().__init__()
        self.encoder = nn.Sequential(
            SAB(head_num, 1, 64, split = split),
            SAB(head_num, 64, 64, split = split)
        )
        self.decoder = nn.Sequential(
            PMA(1, head_num, 64, use_layer_norm, split),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        # encoder outputs (n_batch, x_token, 64)
        X = self.encoder(X) 
        # PMA output (n, 1, 64), after Linear: (n, 1, 1)
        return self.decoder(X).squeeze(-1)
