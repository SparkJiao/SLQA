import torch
import torch.nn as nn
from torch.nn import Linear
from allennlp.modules import TimeDistributed


class FusionLayer(nn.Module):
    def __init__(self, input_dim):
        super(FusionLayer, self).__init__()
        self._input_dim = input_dim
        self._tanh = nn.Tanh()
        self._sigmoid = nn.Sigmoid()
        self._fusion_m = TimeDistributed(Linear(in_features=4 * input_dim, out_features=input_dim))
        self._fusion_g = TimeDistributed(Linear(in_features=4 * input_dim, out_features=1))

    def forward(self, x, y):
        # x: (batch_size, length, encoding_dim*4)
        # y: (batch_size, length, encoding_dim*4)
        # output: (batch_size, length, encoding_dim)
        z = torch.cat((x, y, x * y, x - y), 2)
        return self._sigmoid(self._fusion_g(z)) * self._tanh(self._fusion_m(z)) + (
                1 - self._sigmoid(self._fusion_g(z))) * x
