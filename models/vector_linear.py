import torch
import torch.nn as nn
from torch.nn import Parameter


class VectorLinear(nn.Module):

    def __init__(self, in_features, use_bias=True):
        super(VectorLinear, self).__init__()
        # y = X * w^T + b
        # shape of x: n * in_features
        # shape of w: 1 * in_features
        # shape of y: n * 1
        # shape of b: n * 1
        self._weight_vector = Parameter(torch.Tensor(in_features, 1))
        self._use_bias = use_bias
        if use_bias:
            self.bias = Parameter(torch.Tensor(1))
        self._softmax = torch.nn.Softmax(dim=-1)
        nn.init.xavier_normal(self._weight_vector)


    def forward(self, tensor: torch.Tensor):
        if self._use_bias:
            return self._softmax(torch.mm(tensor, self._weight_vector) + self.bias)
        else:
            return self._softmax(torch.mm(tensor, self._weight_vector))
