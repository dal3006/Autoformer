import math

import torch
import torch.nn as nn


class AutoCorrelation(nn.Module):
    def __init__(self, factor=1):
        super(AutoCorrelation, self).__init__()
        self.factor = factor

    def time_delay_agg(self, V, corr):
        # V:(B, H, L, d_v), corr:(B, H, L, V)
        B, H, L = V.shape[0], V.shape[1], V.shape[2]
        top_k = int(self.factor * math.log(L))
        weights, delays = torch.topk(corr, top_k, dim=2)
        weights = torch.softmax(weights, dim=2)  # (B, H, topK, V)

        init_index = torch.arange(L).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        init_index = init_index.repeat(B, H, 1, V.shape[3]).to(V.device)  # (B, H, L, d_v)
        delays_agg = torch.zeros_like(V).float()
        V = V.repeat(1, 1, 2, 1)

        for i in range(top_k):
            weight = weights[:, :, i, :].unsqueeze(2)  # (B, H, 1, V)
            delay = delays[:, :, i, :].unsqueeze(2)  # (B, H, 1, V)
            index = init_index + delay
            pattern = torch.gather(V, dim=2, index=index)
            delays_agg = delays_agg + pattern * weight
        return delays_agg

    def forward(self, Q, K, V):
        # Q:(B, H, L, d_k), K:(B, H, S, d_k), V:(B, H, S, d_v)
        B, L, S, H = Q.shape[0], Q.shape[2], K.shape[2], K.shape[1]

        # Q:(B, H, L, d_k), K:(B, H, L, d_k), V:(B, H, L, d_v)
        if L > S:
            zeros = torch.zeros_like(Q[:, :, :(L - S), :], device=Q.device).float()
            K = torch.cat([K, zeros], dim=2)
            V = torch.cat([V, zeros], dim=2)
        else:
            V = V[:, :, :L, :]
            K = K[:, :, :L, :]

        q_fft = torch.fft.rfft(Q, dim=2)
        k_fft = torch.fft.rfft(K, dim=2)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=2)  # (B, H, L, V)

        self.time_delay_agg(V, corr)

        return V


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, factor):
        super(AutoCorrelationLayer, self).__init__()
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.correlation = AutoCorrelation(factor)

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads)

        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model)

    def forward(self, Q, K, V):
        # Q:(B, L, d_model), K:(B, S, d_model), V:(B, S, d_model)
        B, L, S, H = Q.shape[0], Q.shape[1], K.shape[1], self.n_heads

        Q = self.W_Q(Q).reshape(B, L, H, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        K = self.W_Q(K).reshape(B, S, H, self.d_k).transpose(1, 2)  # (B, H, S, d_k)
        V = self.W_Q(V).reshape(B, S, H, self.d_v).transpose(1, 2)  # (B, H, S, d_v)

        out = self.correlation(Q, K, V)  # (B, H, L, d_v)
        # out = out.transpose(1, 2)  # (B, L, H, d_v)
        out = out.reshape(B, L, -1)
        out = self.fc(out)
        return out
