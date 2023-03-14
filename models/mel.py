import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Mel2Control(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int = 128, **kwargs
    ):
        super().__init__()

        # conv in stack
        self.stack = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
        )

        # lstm
        self.decoder = nn.LSTM(
            hidden_channels,
            hidden_channels,
            batch_first=True,
            bidirectional=True,
            **kwargs
        )
        self.norm = nn.LayerNorm(hidden_channels * 2)

        # out
        self.dense_out = nn.Linear(hidden_channels * 2, out_channels)
        self.dense_out.weight.data.zero_()
        self.dense_out.bias.data.zero_()

    def forward(self, mels: Tensor):
        x = self.stack(mels).transpose(1, 2)
        x = self.decoder(x)[0]
        x = self.norm(x)
        y = self.dense_out(x)
        return y
