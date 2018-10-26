import torch
from torch import nn

from allennlp.modules import TimeDistributed


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        self._input_dim = input_dim
        self._linear = TimeDistributed(torch.nn.Linear(in_features=input_dim, out_features=input_dim, bias=False))

    def forward(self, d):
        # d: (batch_size, length, encoding_dim)
        # return D W D^T
        return torch.bmm(self._linear(d), d.transpose(2, 1))
