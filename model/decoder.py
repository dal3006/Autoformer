import torch.nn.functional as F
from torch import nn

from model.AutoCorrelation import AutoCorrelationLayer
from model.embed import DataEmbedding
from model.tools import LayerNorm, SeriesDecomp


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, mov_avg, n_heads, factor, dropout):
        super(DecoderLayer, self).__init__()
        self.self_correlation = AutoCorrelationLayer(d_model, n_heads, factor)
        self.cross_correlation = AutoCorrelationLayer(d_model, n_heads, factor)
        self.decomp1 = SeriesDecomp(mov_avg)
        self.decomp2 = SeriesDecomp(mov_avg)
        self.decomp3 = SeriesDecomp(mov_avg)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,), bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,), bias=False)

        self.projection1 = nn.Linear(d_model, d_feature, bias=True)
        self.projection2 = nn.Linear(d_model, d_feature, bias=True)
        self.projection3 = nn.Linear(d_model, d_feature, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, season, enc_out):
        residual = season.clone()
        season = self.dropout(self.self_correlation(season, season, season))
        trend1, season = self.decomp1(season + residual)

        residual = season.clone()
        season = self.dropout(self.cross_correlation(season, enc_out, enc_out))
        trend2, season = self.decomp2(season + residual)

        residual = season.clone()
        season = self.dropout(self.activation(self.conv1(season.permute(0, 2, 1))))
        season = self.dropout(self.conv2(season).permute(0, 2, 1))
        trend3, season = self.decomp3(season + residual)

        trend = self.projection1(trend1) + self.projection2(trend2) + self.projection3(trend3)
        return trend, season


class Decoder(nn.Module):
    def __init__(self, d_feature, d_mark, d_model, d_ff, mov_avg, n_heads, factor, d_layers, dropout, pos):
        super(Decoder, self).__init__()
        self.embedding = DataEmbedding(d_feature, d_mark, d_model, dropout, pos)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, d_feature, d_ff, mov_avg,
                                                          n_heads, factor, dropout) for _ in range(d_layers)])
        self.norm = LayerNorm(d_model)
        self.projection = nn.Linear(d_model, d_feature, bias=True)

    def forward(self, season, trend, dec_mark, enc_out):
        season = self.embedding(season, dec_mark)
        for decoder_layer in self.decoder_layers:
            tmp_trend, season = decoder_layer(season, enc_out)
            trend = trend + tmp_trend

        season = self.norm(season)
        season = self.projection(season)
        return season + trend
