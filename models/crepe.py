import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import AudioTensor


class CREPE(nn.Module):
    hidden_channels = [1024, 128, 128, 128, 256, 512]
    strides = [4, 1, 1, 1, 1, 1]
    grouping = 4
    kernel_sizes = [512, 64, 64, 64, 64, 64]
    hop_length = 256

    def __init__(self, in_channels, out_channels):
        super().__init__()

        in_channels = [in_channels] + self.hidden_channels[:-1]

        self.convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels[i],
                        out_channels=self.hidden_channels[i],
                        kernel_size=self.kernel_sizes[i],
                        stride=self.strides[i],
                        padding=self.kernel_sizes[i] // 2,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.hidden_channels[i]),
                    nn.MaxPool1d(2, 2, 1),
                )
                for i in range(len(self.hidden_channels))
            ]
        )
        self.out_pad = nn.ReflectionPad1d((1, 2))

        self.out_linear = nn.Linear(
            self.hidden_channels[-1] * self.grouping, out_channels
        )

    def forward(self, x: torch.Tensor):
        x = self.out_pad(self.convs(x))
        x = x.unfold(2, self.grouping, 1).permute(0, 2, 1, 3)
        x = self.out_linear(x.reshape(x.shape[0], x.shape[1], -1))
        return AudioTensor(x, hop_length=self.hop_length)
