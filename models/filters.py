import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchaudio.functional import lfilter
from typing import Optional, Union, List, Tuple, Callable, Any


from .lpc import lpc_synthesis
from .utils import (
    get_radiation_time_filter,
    get_window_fn,
    coeff_product,
    complex2biquads,
    params2biquads,
    TimeContext,
)


__all__ = [
    "FilterInterface",
    "LTVMinimumPhaseFilter",
    "LTIRadiationFilter",
    "LTIComplexConjAllpassFilter",
    "LTIRealCoeffAllpassFilter",
]


class FilterInterface(nn.Module):
    def forward(self, ex: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError


class LTVFilterInterface(FilterInterface):
    def forward(self, ex: Tensor, *args, ctx: TimeContext, **kwargs):
        raise NotImplementedError


class LTVMinimumPhaseFilter(LTVFilterInterface):
    def __init__(
        self,
        window: str,
        window_length: int,
    ):
        super().__init__()
        window = get_window_fn(window)(window_length)
        self.register_buffer("_kernel", torch.diag(window).unsqueeze(1))

    def forward(self, ex: Tensor, gain: Tensor, a: Tensor, ctx: TimeContext):
        """
        Args:
            ex (Tensor): [B, T]
            gain (Tensor): [B, T / hop_length]
            a (Tensor): [B, T / hop_length, order]
            ctx (TimeContext): TimeContext
        """

        assert ex.ndim == 2
        assert gain.ndim == 2
        assert a.ndim == 3
        assert a.shape[1] == gain.shape[1]

        hop_length = ctx.hop_length

        window_size = self._kernel.shape[0]
        assert window_size >= hop_length * 2, f"{window_size} < {hop_length * 2}"
        padding = (window_size - hop_length) // 2

        ex = F.pad(
            ex,
            (padding,) * 2,
            "constant",
            0,
        )
        unfolded = ex.unfold(1, window_size, hop_length)
        assert unfolded.shape[1] <= a.shape[1], f"{unfolded.shape} != {a.shape}"
        a = a[:, : unfolded.shape[1]]
        gain = gain[:, : unfolded.shape[1]]

        batch, frames = gain.shape
        unfolded = unfolded.reshape(-1, window_size)
        gain = gain.reshape(-1)
        a = a.reshape(-1, a.shape[-1])
        filtered = lpc_synthesis(unfolded, gain, a).view(batch, frames, -1)

        # overlap-add
        filtered = filtered.transpose(1, 2)
        ones = filtered.new_ones(1, filtered.shape[1], filtered.shape[2])
        tmp = torch.cat([filtered, ones], dim=0)
        tmp = F.conv_transpose1d(
            tmp, self._kernel, stride=hop_length, padding=padding
        ).squeeze(1)

        y = tmp[:-1]
        norm = tmp[-1]

        # normalize
        return y / norm


class LTIRadiationFilter(FilterInterface):
    def __init__(
        self,
        num_zeros: int,
        window: str = "hanning",
    ):
        super().__init__()
        self.register_buffer(
            "_kernel",
            get_radiation_time_filter(num_zeros, get_window_fn(window))
            .flip(0)
            .unsqueeze(0)
            .unsqueeze(0),
        )
        self._padding = self._kernel.size(-1) // 2

    def forward(self, ex: Tensor):
        assert ex.ndim == 2
        return F.conv1d(
            ex.unsqueeze(1),
            self._kernel,
            padding=self._padding,
        ).squeeze(1)


class LTIComplexConjAllpassFilter(FilterInterface):
    max_abs_value: float

    def __init__(self, num_roots: int, max_abs_value: float = 0.99):
        super().__init__()
        self.max_abs_value = max_abs_value
        gain = nn.init.calculate_gain("tanh")
        self.magnitude_logits = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(num_roots), gain=gain)
        )
        self.cos_logits = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(num_roots), gain=gain)
        )

    def forward(self, ex: Tensor):
        assert ex.ndim == 2
        mag = torch.sigmoid(self.magnitude_logits) * self.max_abs_value
        cos = torch.tanh(self.cos_logits)
        sin = torch.sqrt(1 - cos**2)
        roots = mag * (cos + 1j * sin)
        biquads = complex2biquads(roots)
        a_coeffs = coeff_product(biquads.unsqueeze(1)).squeeze()
        b_coeffs = a_coeffs.flip(0)
        return lfilter(ex, a_coeffs, b_coeffs, False)


class LTIRealCoeffAllpassFilter(LTIComplexConjAllpassFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits1 = self.magnitude_logits
        self.logits2 = self.cos_logits
        delattr(self, "magnitude_logits")
        delattr(self, "cos_logits")

    def forward(self, ex: Tensor):
        assert ex.ndim == 2
        biquads = params2biquads(
            self.logits1.tanh() * self.max_abs_value,
            self.logits2.tanh() * self.max_abs_value,
        )
        a_coeffs = coeff_product(biquads.unsqueeze(1)).squeeze()
        b_coeffs = a_coeffs.flip(0)
        return lfilter(ex, a_coeffs, b_coeffs, False)
