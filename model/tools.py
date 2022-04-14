import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, channels):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.norm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class SeriesDecomp(nn.Module):

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.mov_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        trend = self.mov_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        season = x - trend
        return trend, season
