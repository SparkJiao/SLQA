import torch
import torch.nn as nn


class Fusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Fusion, self).__init__()
        self.linear = nn.Linear(input_dim * 4, hidden_dim, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        z = torch.cat([x, y, x * y, x - y], dim=2)
        return self.tanh(self.linear(z))


class Gate(nn.Module):
    def __init__(self, input_dim):
        super(Gate, self).__init__()
        self.linear = nn.Linear(input_dim * 4, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        z = torch.cat([x, y, x * y, x - y], dim=2)
        return self.sigmoid(self.linear(z))


class FusionLayer(nn.Module):
    """
    vector based fusion
    m(x, y) = W([x, y, x * y, x - y]) + b
    g(x, y) = w([x, y, x * y, x - y]) + b
    :returns g(x, y) * m(x, y) + (1 - g(x, y)) * x
    """
    def __init__(self, input_dim):
        super(FusionLayer, self).__init__()
        self.linear_f = nn.Linear(input_dim * 4, input_dim, bias=True)
        self.linear_g = nn.Linear(input_dim * 4, 1, bias=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        z = torch.cat([x, y, x * y, x - y], dim=2)
        gated = self.sigmoid(self.linear_g(z))
        fusion = self.tanh(self.linear_f(z))
        return gated * fusion + (1 - gated) * x

class BilinearSeqAtt(nn.Module):
    def __init__(self, input_dim1, input_dim2):
        super(BilinearSeqAtt, self).__init__()
        self.linear = nn.Linear(input_dim1, input_dim2)

    def forward(self, x, y):
        """
        :param x: b * dim1
        :param y: b * len * dim2
        :return:
        """
        xW = self.linear(x)
        # b * len
        xWy = torch.bmm(y, xW.unsqueeze(2)).squeeze(2)
        return xWy