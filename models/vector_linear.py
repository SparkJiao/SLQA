import torch
import torch.nn as nn
from torch.nn import Parameter


class VectorLinear(nn.Module):

    def __init__(self, in_features, use_bias=True):
        # y = X * w^T + b
        # shape of x: n * in_features
        # shape of w: 1 * in_features
        # shape of y: n * 1
        # shape of b: n * 1
        self._weight_vector = Parameter(torch.Tensor(in_features))
        self._use_bias = use_bias
        if use_bias:
            self.bias = Parameter(torch.Tensor(1))
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        if self._use_bias:
            return self._sigmoid(torch.mm(input, self._weight_vector) + self.bias)
        else:
            return self._sigmoid(torch.mm(input, self._weight_vector))
