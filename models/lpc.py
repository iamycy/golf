import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchaudio.functional import lfilter
from typing import Optional, Union, List, Tuple, Callable


from .utils import get_window_fn


def lpc_synthesis(source: Tensor, gains: Tensor, a: Tensor):
    order = a.shape[-1] + 1
    b = a.new_zeros(a.shape[:-1] + (order,))
    b[..., 0] = gains
    a = torch.cat([a.new_ones(a.shape[:-1] + (1,)), a], dim=-1)
    return lfilter(source, a, b, False)


class LPCSynth(nn.Module):
    def __init__(
        self,
        hop_length: int,
        window_size: int = None,
        window: str = "hann",
    ):
        super().__init__()
        window_fn = get_window_fn(window)
        self.hop_length = hop_length
        self.window_size = hop_length * 4 if window_size is None else window_size
        self.padding = (self.window_size - self.hop_length) // 2
        self.register_buffer(
            "_kernel", torch.diag(window_fn(self.window_size)).unsqueeze(1)
        )

    def forward(self, ex: Tensor, lpc: Tensor):
        assert ex.ndim == 1
        assert lpc.ndim == 2

        ex = F.pad(ex[None, None, :], (self.padding,) * 2, "constant", 0).squeeze()
        unfolded = ex.unfold(0, self.window_size, self.hop_length)
        assert unfolded.shape[0] == lpc.shape[0], f"{unfolded.shape} != {lpc.shape}"

        gain, a = lpc[..., 0], lpc[..., 1:]
        filtered = lpc_synthesis(unfolded, gain, a)

        # overlap-add
        filtered = filtered.t().unsqueeze(0)
        y = F.conv_transpose1d(
            filtered, self._kernel, stride=self.hop_length, padding=self.padding
        ).squeeze()

        # normalize
        ones = torch.ones_like(filtered)
        norm = F.conv_transpose1d(
            ones, self._kernel, stride=self.hop_length, padding=self.padding
        ).squeeze()
        return y / norm


class BatchLPCSynth(LPCSynth):
    def forward(self, ex: Tensor, gain: Tensor, a: Tensor):
        assert ex.ndim == 2
        assert gain.ndim == 2
        assert a.ndim == 3
        assert a.shape[1] == gain.shape[1]

        ex = F.pad(ex[:, None, :], (self.padding,) * 2, "constant", 0).squeeze(1)
        unfolded = ex.unfold(1, self.window_size, self.hop_length)
        assert unfolded.shape[1] <= a.shape[1], f"{unfolded.shape} != {a.shape}"
        a = a[:, : unfolded.shape[1]]
        gain = gain[:, : unfolded.shape[1]]
        
        batch, frames = gain.shape
        unfolded = unfolded.reshape(-1, self.window_size)
        gain = gain.reshape(-1)
        a = a.reshape(-1, a.shape[-1])
        filtered = lpc_synthesis(unfolded, gain, a).view(batch, frames, -1)

        # overlap-add
        filtered = filtered.transpose(1, 2)
        ones = filtered.new_ones(1, filtered.shape[1], filtered.shape[2])
        tmp = torch.cat([filtered, ones], dim=0)
        tmp = F.conv_transpose1d(
            tmp, self._kernel, stride=self.hop_length, padding=self.padding
        ).squeeze(1)

        y = tmp[:-1]
        norm = tmp[-1]

        # normalize
        return y / norm


class BatchSecondOrderLPCSynth(LPCSynth):
    def forward(self, ex: Tensor, gain: Tensor, biquads: Tensor):
        assert ex.ndim == 2
        assert gain.ndim == 2
        assert biquads.ndim == 4 and biquads.shape[-1] == 3

        ex = F.pad(ex[:, None, :], (self.padding,) * 2, "constant", 0).squeeze(1)
        unfolded = ex.unfold(1, self.window_size, self.hop_length)
        assert (
            unfolded.shape[1] <= biquads.shape[1]
        ), f"{unfolded.shape} != {biquads.shape}"
        biquads = biquads[:, : unfolded.shape[1]]
        gain = gain[:, : unfolded.shape[1]]

        batch, frames = gain.shape

        unfolded = unfolded.reshape(-1, self.window_size)
        gain = gain.reshape(-1)
        biquads = biquads.reshape(-1, biquads.shape[-2], biquads.shape[-1])

        unfolded = unfolded * gain[:, None]
        b = torch.zeros_like(biquads[..., 0, :])
        b[..., 0] = 1
        for i in range(biquads.shape[-2]):
            unfolded = lfilter(unfolded, biquads[..., i, :], b, False)

        filtered = unfolded.view(batch, frames, -1).transpose(1, 2)
        ones = filtered.new_ones(1, filtered.shape[1], filtered.shape[2])
        tmp = torch.cat([filtered, ones], dim=0)
        tmp = F.conv_transpose1d(
            tmp, self._kernel, stride=self.hop_length, padding=self.padding
        ).squeeze(1)

        y = tmp[:-1]
        norm = tmp[-1]

        # normalize
        return y / norm
