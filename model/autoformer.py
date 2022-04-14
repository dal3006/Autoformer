import torch
from torch import nn

from model.decoder import Decoder
from model.encoder import Encoder
from model.tools import SeriesDecomp


class Autoformer(nn.Module):
    def __init__(self, pred_len=96, d_feature=7, d_mark=4, d_model=512, d_ff=2048, e_layers=2, d_layers=1, mov_avg=25,
                 n_heads=8, factor=3, dropout=0.05, pos=False):
        super(Autoformer, self).__init__()
        self.pred_len = pred_len

        self.decomp = SeriesDecomp(mov_avg)
        self.encoder = Encoder(d_feature, d_mark, d_model, d_ff, mov_avg, n_heads, factor, e_layers, dropout, pos)
        self.decoder = Decoder(d_feature, d_mark, d_model, d_ff, mov_avg, n_heads, factor, d_layers, dropout, pos)

    def forward(self, enc_x, enc_mark, dec_x, dec_mark):
        enc_out = self.encoder(enc_x, enc_mark)

        mean = torch.mean(enc_x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([dec_x.shape[0], self.pred_len, dec_x.shape[2]], device=dec_x.device)
        trend, season = self.decomp(dec_x[:, :-self.pred_len, :])
        trend = torch.cat([trend, mean], dim=1)
        season = torch.cat([season, zeros], dim=1)

        dec_out = self.decoder(season, trend, dec_mark, enc_out)

        return dec_out[:, -self.pred_len:, :]
