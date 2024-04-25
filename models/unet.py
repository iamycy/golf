import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram
from typing import Tuple, Dict, Union, Callable, List
from functools import reduce
import math

from .audiotensor import AudioTensor
from .enc import BackboneModelInterface
from .lru import LRU


def interp_env(x: torch.Tensor, dyn_freqs: torch.Tensor, static_freqs: torch.Tensor):
    assert x.shape == dyn_freqs.shape

    # x: (batch, freq_dyn)
    # dyn_freqs: (batch, freq_dyn)
    # static_freqs: (freq_static)

    intervals = dyn_freqs[:, 1:] - dyn_freqs[:, :-1]
    tmp = static_freqs - dyn_freqs[:, :-1, None]
    mask = (0 <= tmp) & (tmp < intervals[:, :, None])
    valid_mask = mask.any(1)
    low_index = torch.argmax(mask.long(), 1).clip(0, mask.shape[1] - 1)
    low_freqs = dyn_freqs.gather(1, low_index)
    p = (static_freqs - low_freqs) / intervals.gather(1, low_index)
    env = x.gather(1, low_index) * (1 - p) + x.gather(1, low_index + 1) * p
    return torch.where(valid_mask, env, 0)


class LRUBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0,
        mlp_factor: int = 4,
    ):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size, bias=False)
        self.norm = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )
        self.lru = nn.ModuleList(
            [LRU(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * mlp_factor),
                    nn.GELU(),
                    nn.Linear(hidden_size * mlp_factor, hidden_size),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            reduce(
                lambda h, m: h + m[0](m[1](m[2](h))[0]),
                zip(self.ff, self.lru, self.norm),
                self.proj(x),
            ),
            None,
        )


class UNetEncoder(BackboneModelInterface):
    def __init__(
        self,
        out_channels: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        channels: List = [16, 32, 64, 128],
        strides: List = [4, 4, 4, 4],
        lstm_hidden_size: int = 128,
        include_env_features: bool = False,
        num_harmonics: int = 150,
        sample_rate: int = 22050,
        f0_conditioning: bool = True,
        use_lru: bool = False,
        **lstm_kwargs,
    ):
        super().__init__(
            lstm_hidden_size if use_lru else lstm_hidden_size * 2, out_channels
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectrogram = Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True)
        self.f0_conditioning = f0_conditioning

        in_channels = 1 if not include_env_features else 4
        blocks = []
        for out_channels, stride in zip(channels, strides):
            blocks.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        (stride * 2 + 1, 3),
                        padding=(stride, 1),
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d((stride, 1), stride=(stride, 1)),
                ]
            )
            in_channels = out_channels

        flatten_size = (
            (n_fft // 2 + 1) // reduce(lambda x, y: x * y, strides) * in_channels
        )

        self.cnns = nn.Sequential(*blocks)
        if not use_lru:
            self.lstm = nn.LSTM(
                flatten_size + 1 if f0_conditioning else flatten_size,
                lstm_hidden_size,
                batch_first=True,
                bidirectional=True,
                **lstm_kwargs,
            )
            self.norm = nn.LayerNorm(lstm_hidden_size * 2)
        else:
            self.lstm = LRUBlock(
                flatten_size + 1 if f0_conditioning else flatten_size,
                lstm_hidden_size,
                **lstm_kwargs,
            )
            self.norm = nn.LayerNorm(lstm_hidden_size)

        self.register_buffer("log_spec_min", torch.tensor(torch.inf))
        self.register_buffer("log_spec_max", torch.tensor(-torch.inf))

        self.include_env_features = include_env_features
        self.num_harmonics = num_harmonics
        self.sr = sample_rate

    def forward(self, x: AudioTensor, f0: AudioTensor = None) -> AudioTensor:
        assert x.hop_length == 1
        spec = self.spectrogram(x.as_tensor())
        if self.f0_conditioning and f0 is not None:
            f0 = f0.set_hop_length(self.hop_length).truncate(spec.size(2)).as_tensor()
            spec = spec[..., : f0.size(-1)]
        if self.include_env_features and self.f0_conditioning:
            spec = spec.mT
            intervals = self.sr / self.n_fft
            freqs = torch.arange(0, self.n_fft // 2 + 1, device=spec.device) * intervals
            f0_full = torch.where(f0 > 0, f0, self.sr / 2 / (self.num_harmonics - 1))
            pickup_freqs = f0_full.unsqueeze(-1) * torch.arange(
                0.0, self.num_harmonics + 1, 0.5, device=spec.device
            )
            indexes = (
                (pickup_freqs / intervals).round().long().clip(0, spec.shape[2] - 1)
            )
            energies = spec.gather(2, indexes)
            harms_energy = energies[..., ::2]
            noise_energy = torch.cat([energies[..., :1], energies[..., 1::2]], -1)

            remap_indexs = freqs / f0_full.unsqueeze(-1)
            remap_indexs_low = (
                remap_indexs.floor().long().clip(0, self.num_harmonics - 2)
            )
            p = remap_indexs - remap_indexs_low
            p = p.clip(0, 1)
            harm_env = (1 - p) * harms_energy.gather(
                2, remap_indexs_low
            ) + p * harms_energy.gather(2, remap_indexs_low + 1)

            remap_indexs = (freqs + f0_full.unsqueeze(-1) * 0.5) / f0_full.unsqueeze(-1)
            remap_indexs_low = (
                remap_indexs.floor().long().clip(0, self.num_harmonics - 2)
            )
            p = remap_indexs - remap_indexs_low
            p[remap_indexs_low == 0] = (p[remap_indexs_low == 0] - 0.5) * 2
            p = p.clip(0, 1)
            noise_env = (1 - p) * noise_energy.gather(
                2, remap_indexs_low
            ) + p * noise_energy.gather(2, remap_indexs_low + 1)

            harm_env = torch.maximum(harm_env, noise_env)

            spec = torch.stack([spec, harm_env, noise_env], dim=1).transpose(2, 3)
            snr = (noise_env / (harm_env + noise_env + 1e-16)).mT.unsqueeze(1) * 2
        else:
            spec = spec.unsqueeze(1)

        log_spec = spec.add(1e-8).log()

        if self.training:
            self.log_spec_min.fill_(min(self.log_spec_min, torch.min(log_spec).item()))
            self.log_spec_max.fill_(max(self.log_spec_max, torch.max(log_spec).item()))
        feature = (log_spec - self.log_spec_min) / (
            self.log_spec_max - self.log_spec_min
        )

        if self.include_env_features:
            feature = torch.cat([feature, snr], dim=1)

        h = self.cnns(feature)
        h = torch.flatten(h, 1, 2).mT
        if self.f0_conditioning and f0 is not None:
            h = torch.cat([h, torch.log1p(f0).unsqueeze(-1)], dim=-1)
        h = self.lstm(h)[0]
        h = self.norm(h)
        return AudioTensor(super().forward(h), hop_length=self.hop_length)


class UNetEncoderV2(BackboneModelInterface):
    def __init__(
        self,
        sr: int,
        out_channels: int,
        embed_size: int = 8,
        n_fft: int = 1024,
        hop_length: int = 256,
        channels: List = [16, 32, 64, 128],
        strides: List = [4, 4, 4, 4],
        lstm_hidden_size: int = 128,
        **lstm_kwargs,
    ):
        super().__init__(lstm_hidden_size * 2, out_channels)
        self.sr = sr
        self.embeddings = nn.Embedding(2, embed_size)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectrogram = Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True)

        in_channels = 1 + embed_size
        blocks = []
        for out_channels, stride in zip(channels, strides):
            blocks.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        (stride * 2 + 1, 3),
                        padding=(stride, 1),
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d((stride, 1), stride=(stride, 1)),
                ]
            )
            in_channels = out_channels

        flatten_size = (
            (n_fft // 2 + 1) // reduce(lambda x, y: x * y, strides) * in_channels
        )

        self.cnns = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(
            flatten_size + 1,
            lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
            **lstm_kwargs,
        )
        self.norm = nn.LayerNorm(lstm_hidden_size * 2)

        self.register_buffer("log_spec_min", torch.tensor(torch.inf))
        self.register_buffer("log_spec_max", torch.tensor(-torch.inf))

    def forward(self, x: AudioTensor, f0: AudioTensor) -> AudioTensor:
        log_spec = self.spectrogram(x.as_tensor().unsqueeze(1)).add(1e-8).log()
        if self.training:
            self.log_spec_min.fill_(min(self.log_spec_min, torch.min(log_spec).item()))
            self.log_spec_max.fill_(max(self.log_spec_max, torch.max(log_spec).item()))
        feature = (log_spec - self.log_spec_min) / (
            self.log_spec_max - self.log_spec_min
        )

        f0 = (
            f0.set_hop_length(self.hop_length)
            .truncate(feature.size(-1))
            .as_tensor()
            .unsqueeze(2)
        )
        feature = feature[..., : f0.size(1)]

        # calculate harmonics mask
        freqs = (
            torch.arange(
                0,
                feature.size(-2),
                device=f0.device,
            )
            * self.sr
            / self.n_fft
        )
        harms_index = freqs / f0
        harms_mask = harms_index % 1
        harms_mask = (harms_mask < 0.25) | (harms_mask > 0.75)
        harms_mask[harms_index <= 0.75] = False
        harms_embed = self.embeddings(harms_mask.long()).permute(0, 3, 2, 1)
        feature = torch.cat([feature, harms_embed], dim=1)

        h = self.cnns(feature)
        h = torch.flatten(h, 1, 2).mT
        h = torch.cat([h, torch.log1p(f0)], dim=-1)
        h = self.lstm(h)[0]
        h = self.norm(h)
        return AudioTensor(
            super().forward(h), hop_length=self.hop_length * x.hop_length
        )


def sinusoidal(
    min_scale: float = 1.0,
    max_scale: float = 10000.0,
    shape: tuple = (512, 512),
    permute_bands: bool = False,
    random_phase_offsets: bool = False,
):
    """Creates 1D Sinusoidal Position Embedding Initializer.

    Args:
            min_scale: Minimum frequency-scale in sine grating.
            max_scale: Maximum frequency-scale in sine grating.
            dtype: The DType of the returned values.
            permute_bands: If True, sinusoid band order will be randomly permuted at
            initialization.
            random_phase_offsets: If True, each band's phase will have a random offset
            at initialization.

    Returns:
            The sinusoidal initialization function.
    """
    max_len, features = shape
    position = torch.arange(0, max_len).unsqueeze(1)
    scale_factor = -math.log(max_scale / min_scale) / (features // 2 - 1)
    div_term = min_scale * torch.exp(torch.arange(0, features // 2) * scale_factor)
    rads = position * div_term
    if random_phase_offsets:
        sin_offsets = torch.rand(features // 2) * 2 * math.pi
        cos_offsets = torch.rand(features // 2) * 2 * math.pi
    else:
        sin_offsets = 0.0
        cos_offsets = 0.0
    pe = torch.empty(max_len, features, dtype=rads.dtype)
    pe[:, : features // 2] = torch.sin(rads + sin_offsets)
    pe[:, features // 2 :] = torch.cos(rads + cos_offsets)
    if permute_bands:
        pe = pe[:, torch.randperm(features)]
    return pe


class TransformerEncoder(BackboneModelInterface):
    def __init__(
        self,
        out_channels: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        emb_channels: int = 32,
        kernel_size: Tuple[int, int] = (5, 3),
        stride: int = 2,
        maxpool_stride: int = 64,
        nhead: int = 4,
        num_attn_layers: int = 4,
        lstm_hidden_size: int = 128,
        dropout: float = 0.1,
        **lstm_kwargs,
    ):
        super().__init__(lstm_hidden_size * 2, out_channels)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectrogram = Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True)

        self.conv = nn.Sequential(
            nn.Conv2d(
                1,
                emb_channels,
                kernel_size,
                stride=(stride, 1),
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            ),
            nn.BatchNorm2d(emb_channels),
            nn.LeakyReLU(0.2),
        )

        seq_length = (n_fft // 2 + 1) // stride

        self.register_buffer(
            "positional_embedding", sinusoidal(shape=(seq_length, emb_channels))
        )

        self.attn_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_channels,
                nhead=nhead,
                dim_feedforward=emb_channels * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_attn_layers,
            norm=nn.LayerNorm(emb_channels),
        )

        self.maxpool = nn.MaxPool2d((maxpool_stride, 1), stride=(maxpool_stride, 1))
        self.reduce_seq_length = seq_length // maxpool_stride

        self.lstm = nn.LSTM(
            emb_channels * self.reduce_seq_length + 1,
            lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            **lstm_kwargs,
        )
        self.norm = nn.LayerNorm(lstm_hidden_size * 2)

        self.register_buffer("log_spec_min", torch.tensor(torch.inf))
        self.register_buffer("log_spec_max", torch.tensor(-torch.inf))

    def forward(self, x: AudioTensor, f0: AudioTensor) -> AudioTensor:
        assert x.hop_length == 1
        log_spec = self.spectrogram(x.as_tensor().unsqueeze(1)).add(1e-8).log()
        if self.training:
            self.log_spec_min.fill_(min(self.log_spec_min, torch.min(log_spec).item()))
            self.log_spec_max.fill_(max(self.log_spec_max, torch.max(log_spec).item()))
        feature = (log_spec - self.log_spec_min) / (
            self.log_spec_max - self.log_spec_min
        )

        feature = self.conv(feature)
        B, _, F, T = feature.shape
        feature = feature.permute(0, 3, 2, 1).flatten(0, 1) + self.positional_embedding
        feature = self.attn_enc(feature)

        feature = feature.reshape(B, T, F, -1).permute(0, 3, 2, 1)
        feature = self.maxpool(feature)
        feature = torch.flatten(feature, 1, 2).mT

        f0 = (
            f0.set_hop_length(self.hop_length)
            .truncate(feature.size(1))
            .as_tensor()
            .unsqueeze(2)
        )
        feature = feature[:, : f0.size(1)]
        feature = torch.cat([feature, torch.log1p(f0)], dim=-1)
        feature = self.lstm(feature)[0]
        feature = self.norm(feature)
        return AudioTensor(super().forward(feature), hop_length=self.hop_length)
