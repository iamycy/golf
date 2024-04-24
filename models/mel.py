import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram

from .enc import BackboneModelInterface
from .audiotensor import AudioTensor


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
        return mels.new_tensor(super().forward(x))


class X2Control(Mel2Control):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, **kwargs):
        super().__init__(in_channels=n_fft // 2 + 2, **kwargs)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectrogram = Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True)

        self.register_buffer("log_spec_min", torch.tensor(torch.inf))
        self.register_buffer("log_spec_max", torch.tensor(-torch.inf))

    def forward(self, x: AudioTensor, f0: AudioTensor) -> AudioTensor:
        log_spec = self.spectrogram(x.as_tensor()).add(1e-8).log().mT
        if self.training:
            self.log_spec_min.fill_(min(self.log_spec_min, torch.min(log_spec).item()))
            self.log_spec_max.fill_(max(self.log_spec_max, torch.max(log_spec).item()))
        h = (log_spec - self.log_spec_min) / (self.log_spec_max - self.log_spec_min)

        f0 = (
            f0.set_hop_length(self.hop_length)
            .truncate(h.size(1))
            .as_tensor()
            .unsqueeze(2)
        )
        h = h[:, : f0.size(1)]
        h = torch.cat([h, torch.log1p(f0)], dim=-1)
        h = AudioTensor(h, hop_length=self.hop_length)
        return super().forward(h)


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


@torch.jit.script
def fused_gate(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1.tanh() * x2.sigmoid()


class NonCausalLayer(nn.Module):
    def __init__(
        self,
        radix,
        dilation,
        residual_channels,
        last_layer=False,
    ):
        super().__init__()
        pad_size = dilation * (radix - 1) // 2
        self.W = nn.Conv1d(
            residual_channels,
            residual_channels * 2,
            kernel_size=radix,
            padding=pad_size,
            dilation=dilation,
        )

        self.chs_split = [residual_channels]
        if last_layer:
            self.W_o = nn.Conv1d(residual_channels, residual_channels, 1)
        else:
            self.W_o = nn.Conv1d(residual_channels, residual_channels * 2, 1)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x):
        zw, zf = self.W(x).chunk(2, 1)
        z = fused_gate(zw, zf)
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        return z[0] + x if len(z) else None, skip


class WN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        residual_channels=128,
        depth=20,
        cycle=6,
        radix=3,
    ):
        super().__init__()
        dilations = [2 ** (i % cycle) for i in range(depth)]
        self.dilations = dilations
        self.in_chs = in_channels
        self.res_chs = residual_channels
        self.rdx = radix
        self.r_field = sum(self.dilations) + 1

        self.start = nn.Conv1d(in_channels, residual_channels, 1)

        self.layers = nn.ModuleList(
            NonCausalLayer(radix, d, residual_channels) for d in self.dilations[:-1]
        )
        self.layers.append(
            NonCausalLayer(
                radix,
                self.dilations[-1],
                residual_channels,
                last_layer=True,
            )
        )

        self.end = nn.Conv1d(residual_channels, out_channels, 1)

    def forward(self, x):
        x = self.start(x)
        cum_skip = 0
        for layer in self.layers:
            x, skip = layer(x)
            cum_skip = cum_skip + skip
        return self.end(cum_skip)
