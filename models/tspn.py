import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple, Callable


class TopNGenerator(nn.Module):
    def __init__(self, num_emb, key_emb_size, value_emb_size):
        super().__init__()
        self.key_dict = nn.Parameter(torch.randn(num_emb, key_emb_size))
        self.value_dict = nn.Parameter(torch.randn(num_emb, value_emb_size))

    def forward(self, query, top_n):
        """
        Args:
            query: (batch_size, key_emb_size)
            top_n: int
        """
        # compute cosine similarity
        prod = query @ self.key_dict.t()
        norm = query.norm(dim=-1).unsqueeze(-1) * self.key_dict.norm(dim=-1)
        cos = prod / F.threshold(norm, 1e-8, 1e-8)
        # get top_n indices
        weights, indices = cos.topk(top_n, dim=-1)
        # gather values
        values = self.value_dict[indices] * weights.unsqueeze(-1)
        return values


class TTSPNEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, lstm_channels, **kwargs):
        super().__init__(d_model=d_model, batch_first=True, activation="gelu", **kwargs)

        self.linear3 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_channels,
            batch_first=True,
            bidirectional=True,
        )
        self.linear4 = nn.Linear(lstm_channels * 2, d_model)

    def forward(
        self,
        src: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        # src: (batch_size, frames, K, d_model)
        x = src.view(-1, src.shape[-2], src.shape[-1])
        u = super().forward(x, *args, **kwargs)
        u = u.view(src.shape[0], src.shape[1], src.shape[2], -1)
        v = self.linear4(self.lstm(self.linear3(u).sum(dim=-2))[0])
        return u + v.unsqueeze(-2)


class TTSPNEncoder(nn.TransformerEncoder):
    def __init__(self, out_channels, num_layers, d_model, **kwargs):
        super().__init__(
            TTSPNEncoderLayer(d_model=d_model, **kwargs), num_layers=num_layers
        )
        self.linear = nn.Linear(d_model, out_channels)

    def forward(self, src: Tensor) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output)
        return self.linear(output)
