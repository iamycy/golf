import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .enc import BackboneModelInterface
from .utils import AudioTensor


class Mel2Control(BackboneModelInterface):
    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int = 128, **kwargs
    ):
        super().__init__(hidden_channels * 2, out_channels)

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

    def forward(self, mels: AudioTensor):
        x = torch.transpose(self.stack(torch.transpose(mels, 1, 2)), 1, 2)
        x = self.decoder(x)[0]
        x = self.norm(x)
        return super().forward(x)


class LPCFrameNet(BackboneModelInterface):
    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int = 128, **kwargs
    ):
        super().__init__(hidden_channels, out_channels)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=1),
            nn.Tanh(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.Tanh(),
        )

        self.fc = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.Tanh())

    def forward(self, mels: AudioTensor):
        x = torch.transpose(self.cnn(torch.transpose(mels, 1, 2)), 1, 2)
        x = self.fc(x)
        return super().forward(x)
