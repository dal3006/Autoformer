import math

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = self.pe[: x.size(1), :]
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, d_feature, d_model):
        super(TokenEmbedding, self).__init__()
        self.embed = nn.Conv1d(in_channels=d_feature, out_channels=d_model, kernel_size=(1,))

    def forward(self, x):
        return self.embed(x.transpose(1, 2)).transpose(1, 2)


class TimeEmbedding(nn.Module):
    def __init__(self, d_mark, d_model):
        super(TimeEmbedding, self).__init__()
        self.embed = nn.Linear(d_mark, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, d_feature, d_mark, d_model, dropout=0.1, pos=False):
        super(DataEmbedding, self).__init__()
        self.pos = pos

        self.value_embedding = TokenEmbedding(d_feature=d_feature, d_model=d_model)
        self.time_embedding = TimeEmbedding(d_mark=d_mark, d_model=d_model)

        if self.pos:
            self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if self.pos:
            x = self.value_embedding(x) + self.position_embedding(x) + self.time_embedding(x_mark)
        else:
            x = self.value_embedding(x) + self.time_embedding(x_mark)

        return self.dropout(x)
