import torch
from torch import nn, Tensor
import math
from importlib import import_module
from typing import Optional, Union, List, Tuple, Callable, Any, Dict
from torchaudio.transforms import Spectrogram
from itertools import accumulate, tee

from .utils import get_logits2biquads, biquads2lpc, rc2lpc, get_window_fn
from .audiotensor import AudioTensor

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class BackboneModelInterface(nn.Module):
    # out_channels: int
    # last_layer_channels: int = None
    out_linear: nn.Linear

    def __init__(self, linear_in_channels: int, linear_out_channels: int):
        super().__init__()
        self.out_linear = nn.Linear(linear_in_channels, linear_out_channels)
        self.out_linear.weight.data.zero_()
        self.out_linear.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.out_linear(x)


class VocoderParameterEncoderInterface(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        learn_voicing: bool = False,
        learn_f0: bool = True,
        f0_min: float = 80,
        f0_max: float = 1000,
        split_sizes: Tuple[Tuple[int, ...], ...] = (),
        trsfms: Tuple[Callable[..., Tuple[torch.Tensor, ...]], ...] = (),
        args_keys: Tuple[str, ...] = (),
        **kwargs,
    ):
        super().__init__()

        append_before = lambda x, cond, y: (y,) + x if cond else x
        append_one = lambda x, cond: append_before(x, cond, (1,))

        self.split_sizes = append_one(append_one(split_sizes, learn_voicing), learn_f0)
        self.trsfms = append_before(
            append_before(trsfms, learn_voicing, lambda x: x),
            learn_f0,
            lambda logits: torch.exp(
                torch.sigmoid(logits) * (math.log(f0_max) - math.log(f0_min))
                + math.log(f0_min)
            ),
        )
        self.args_keys = append_before(
            append_before(args_keys, learn_voicing, "voicing_logits"),
            learn_f0,
            "f0",
        )

        module_path, class_name = backbone_type.rsplit(".", 1)
        module = import_module(module_path)

        self.backbone = getattr(module, class_name)(
            out_channels=sum(sum(self.split_sizes, ())), **kwargs
        )
        # self.backbone = torch.compile(self.backbone)

    def forward(
        self, x: AudioTensor, *args: Any, **kwargs: Any
    ) -> Dict[str, Union[AudioTensor, Tuple[AudioTensor]]]:
        h = self.backbone(x, *args, **kwargs)
        logits = [
            h.new_tensor(torch.squeeze(t, 2))
            for t in torch.split(h, sum(self.split_sizes, ()), dim=2)
        ]
        params = dict(
            zip(
                self.args_keys,
                map(
                    lambda f, args: f(*args),
                    self.trsfms,
                    (
                        logits[i:j]
                        for i, j in pairwise(
                            accumulate(
                                map(lambda x: len(x), self.split_sizes), initial=0
                            )
                        )
                    ),
                ),
            )
        )

        return params


class F0EnergyEncoder(BackboneModelInterface):
    def __init__(
        self,
        out_channels: int,
        sr: int = 24000,
        n_fft: int = 2048,
        win_length: int = 960,
        window: str = "hanning",
        hop_length: int = 240,
        num_bands: int = 150,
        lstm_hidden_size: int = 128,
        **lstm_kwargs,
    ):
        super().__init__(lstm_hidden_size * 2, out_channels)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.num_bands = num_bands
        self.freq_interval = sr / n_fft
        self.spectrogram = Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            window_fn=get_window_fn(window),
            hop_length=hop_length,
            center=True,
        )

        self.lstm = nn.LSTM(
            num_bands * 2 + 1,
            lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
            **lstm_kwargs,
        )
        self.norm = nn.LayerNorm(lstm_hidden_size * 2)

        self.register_buffer("log_energy_min", torch.tensor(torch.inf))
        self.register_buffer("log_energy_max", torch.tensor(-torch.inf))

    def forward(self, x: AudioTensor, f0: AudioTensor) -> AudioTensor:
        assert x.hop_length == 1
        spec = self.spectrogram(x.as_tensor()).mT
        spec[..., -1] = 0
        f0 = f0.set_hop_length(self.hop_length).truncate(spec.size(1)).as_tensor()
        spec = spec[:, : f0.size(1)]
        f0_nonzero = torch.where(f0 > 0, f0, self.sr / self.num_bands * 0.5)
        harms = f0_nonzero.unsqueeze(-1) * torch.arange(
            1, self.num_bands + 0.5, 0.5, device=f0.device
        )
        harms = torch.cat([harms[..., :1] * 0.5, harms], dim=-1)
        harms_index = (
            torch.round(harms / self.freq_interval).long().clip(0, spec.size(-1) - 1)
        )
        flatten_indexes = (
            (
                torch.arange(
                    spec.shape[0], device=harms_index.device, dtype=harms_index.dtype
                ).unsqueeze(-1)
                * spec.shape[1]
                + torch.arange(
                    spec.shape[1], device=harms_index.device, dtype=harms_index.dtype
                )
            ).unsqueeze(-1)
            * spec.shape[2]
            + harms_index
        ).flatten()
        harms_energy = spec.flatten()[flatten_indexes].reshape(*harms_index.shape)
        log_energy = torch.log(harms_energy + 1e-8)
        if self.training:
            self.log_energy_min.fill_(
                min(self.log_energy_min, torch.min(log_energy).item())
            )
            self.log_energy_max.fill_(
                max(self.log_energy_max, torch.max(log_energy).item())
            )
        feature = (log_energy - self.log_energy_min) / (
            self.log_energy_max - self.log_energy_min
        )
        log_f0 = torch.log(f0_nonzero)
        feature = torch.cat([feature, log_f0.unsqueeze(-1)], dim=-1)

        h = self.lstm(feature)[0]
        h = self.norm(h)
        return AudioTensor(super().forward(h), hop_length=self.hop_length)
