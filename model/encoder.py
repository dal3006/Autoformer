import torch.nn.functional as F
from torch import nn

from model.AutoCorrelation import AutoCorrelationLayer, AutoCorrelation
from model.embed import DataEmbedding
from model.tools import LayerNorm, SeriesDecomp


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, mov_avg, n_heads, factor, dropout):
        super(EncoderLayer, self).__init__()
        self.self_correlation = AutoCorrelationLayer(d_model, n_heads, factor)
        self.decomp1 = SeriesDecomp(mov_avg)
        self.decomp2 = SeriesDecomp(mov_avg)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,), bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,), bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        residual = x.clone()
        x = self.dropout(self.self_correlation(x, x, x))
        _, x = self.decomp1(x + residual)

        residual = x.clone()
        x = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        _, x = self.decomp2(x + residual)
        return x


class Encoder(nn.Module):
    def __init__(self, d_feature, d_mark, d_model, d_ff, mov_avg, n_heads, factor, e_layers, dropout, pos):
        super(Encoder, self).__init__()
        self.embedding = DataEmbedding(d_feature, d_mark, d_model, dropout, pos)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, d_ff, mov_avg,
                                                          n_heads, factor, dropout) for _ in range(e_layers)])
        self.norm = LayerNorm(d_model)

    def forward(self, enc_x, enc_mark):
        x = self.embedding(enc_x, enc_mark)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = self.norm(x)
        return x
