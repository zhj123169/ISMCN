import torch
import torch.nn as nn


# from math import sqrt

class time2vec(nn.Module):
    """
    Encode Timestamp to a Vector
      Input shape
        - 2D tensor with shape: ``(batch_size, event_num)``.
      Output shape
        - 3D tensor with shape: ``(batch_size, event_num, embedding_size)``.
    """

    def __init__(self, out_features, in_features=1):
        super(time2vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, x):
        x = x.unsqueeze(-1)
        v1 = self.f(torch.matmul(x, self.w) + self.b)
        v2 = torch.matmul(x, self.w0) + self.b0
        return torch.cat([v1, v2], 2)
