import torch
import torch.nn as nn
from torch.nn import Parameter


class VectorMatrixLinear(nn.Module):

    def __init__(self, in_featrues: int, out_features: int):
        self._in_featrues = in_featrues
        self._out_features = out_features
        pass
        self._weight
        self._bias
        self._sigmoid = nn.Sigmoid()

    def forward(self, vector: torch.Tensor, matrix: torch.Tensor):
        pass

