import torch
from torch import Tensor
import numpy as np
import pyworld as pw
from functools import partial
import torch.nn.functional as F
import math
from scipy.signal import get_window
from typing import Any, Callable, Optional, Tuple, Union, List


class AudioTensor(object):
    def __init__(
        self,
        data: Union[Tensor, np.ndarray],
        hop_length: int = 1,
        **kwargs,
    ):
        self._data = torch.as_tensor(data, **kwargs)
        self.hop_length = hop_length if self._data.ndim > 1 else 9223372036854775807

    def __repr__(self):
        return f"Hop-length: {self.hop_length}\n" + repr(self._data)

    def __getitem__(self, index):
        return AudioTensor(self._data[index], hop_length=self.hop_length)

    def unfold(self, size: int, step: int = 1):
        assert self.ndim == 2
        return AudioTensor(
            self._data.unfold(1, size, step), hop_length=self.hop_length * step
        )

    @property
    def shape(self):
        return self._data.shape

    @property
    def names(self):
        return self._data.names

    @property
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def size(self):
        return self._data.size

    @property
    def ndim(self):
        return self._data.ndim

    def dim(self):
        return self._data.dim()

    def __matmul__(self, other):
        return torch.matmul(self, other)

    def __rmatmul__(self, other):
        return torch.matmul(other, self)

    def __neg__(self):
        return torch.neg(self)

    def __add__(self, other):
        return torch.add(self, other)

    def __sub__(self, other):
        return torch.sub(self, other)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __truediv__(self, other):
        return torch.div(self, other)

    def __floordiv__(self, other):
        return torch.floor_divide(self, other)

    def __mod__(self, other):
        return torch.remainder(self, other)

    def __radd__(self, other):
        return torch.add(other, self)

    def __rsub__(self, other):
        return torch.sub(other, self)

    def __rmul__(self, other):
        return torch.mul(other, self)

    def __rtruediv__(self, other):
        return torch.div(other, self)

    def __rfloordiv__(self, other):
        return torch.floor_divide(other, self)

    def __rmod__(self, other):
        return torch.remainder(other, self)

    def __lt__(self, other):
        return torch.lt(self, other)

    def __le__(self, other):
        return torch.le(self, other)

    def __gt__(self, other):
        return torch.gt(self, other)

    def __ge__(self, other):
        return torch.ge(self, other)

    def __eq__(self, other):
        return torch.eq(self, other)

    def __ne__(self, other):
        return torch.ne(self, other)

    def set_hop_length(self, hop_length: int):
        assert hop_length > 0, "hop_length must be positive"
        if hop_length > self.hop_length:
            assert hop_length % self.hop_length == 0
            return self.increase_hop_length(hop_length // self.hop_length)
        elif hop_length < self.hop_length:
            assert self.hop_length % hop_length == 0
            return self.reduce_hop_length(self.hop_length // hop_length)
        return self

    def increase_hop_length(self, factor: int):
        assert factor > 0, "factor must be positive"
        if factor == 1 or self.ndim < 2:
            return self

        data = self._data[:, ::factor]
        return AudioTensor(data, hop_length=self.hop_length * factor)

    def reduce_hop_length(self, factor: int = None):
        if factor is None:
            factor = self.hop_length
        else:
            assert self.hop_length % factor == 0 and factor <= self.hop_length

        if factor == 1 or self.ndim < 2:
            return self

        self_copy = self._data
        # swap the time dimension to the last
        if self.ndim > 2:
            self_copy = self_copy.transpose(1, -1)
        ctx = TimeContext(factor)
        expand_self_copy = linear_upsample(ctx, self_copy)

        # swap the time dimension back
        if self.ndim > 2:
            expand_self_copy = expand_self_copy.transpose(1, -1)

        return AudioTensor(expand_self_copy, hop_length=self.hop_length // factor)

    @property
    def steps(self):
        if self.ndim < 2:
            return 1
        return self._data.size(1)

    def truncate(self, steps: int):
        if steps >= self.steps:
            return self
        data = self._data.narrow(1, 0, steps)
        return AudioTensor(data, hop_length=self.hop_length)

    def as_tensor(self):
        return self._data

    def new_tensor(self, data: Tensor):
        return AudioTensor(data, hop_length=self.hop_length)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in (
            torch.add,
            torch.mul,
            torch.div,
            torch.sub,
            torch.floor_divide,
            torch.remainder,
            torch.lt,
            torch.le,
            torch.gt,
            torch.ge,
            torch.eq,
            torch.ne,
            torch.where,
            torch.matmul,
        ):
            audio_tensors = tuple(a for a in args if isinstance(a, AudioTensor))
            audio_tensors = AudioTensor.broadcasting(*audio_tensors)
            min_steps = min(a.steps for a in audio_tensors)
            audio_tensors = tuple(a.truncate(min_steps) for a in audio_tensors)
            broadcasted_args = []
            i = 0
            for a in args:
                if isinstance(a, AudioTensor):
                    broadcasted_args.append(audio_tensors[i])
                    i += 1
                else:
                    broadcasted_args.append(a)
            args = broadcasted_args
        elif func in (torch.cat, torch.stack):
            raise NotImplementedError(
                "AudioTensors do not support torch.cat and torch.stack"
            )
        if kwargs is None:
            kwargs = {}
        hop_lengths = []
        for a in args:
            if isinstance(a, AudioTensor):
                hop_lengths.append(a.hop_length)
            elif isinstance(a, (tuple, list)):
                for aa in a:
                    if isinstance(aa, AudioTensor):
                        hop_lengths.append(aa.hop_length)

        assert len(hop_lengths) > 0 and all(
            h == hop_lengths[0] for h in hop_lengths
        ), "All AudioTensors must have the same hop length but got {}".format(
            ", ".join(str(h) for h in hop_lengths)
        )
        args = tuple(a.as_tensor() if isinstance(a, AudioTensor) else a for a in args)
        ret = func(*args, **kwargs)
        if isinstance(ret, torch.Tensor) and ret.ndim != 0 and len(hop_lengths) > 0:
            return AudioTensor(ret, hop_length=hop_lengths[0])
        return ret

    @classmethod
    def broadcasting(cls, *tensors):
        assert len(tensors) > 0
        # check hop lengths are divisible by each other
        hop_lengths = tuple(t.hop_length for t in tensors)
        minimum_hop_length = min(hop_lengths)
        assert all(
            h % minimum_hop_length == 0 for h in hop_lengths
        ), "All hop lengths must be divisible by each other"
        ret = tuple(
            t.reduce_hop_length(t.hop_length // minimum_hop_length)
            if t.hop_length > minimum_hop_length
            else t
            for t in tensors
        )
        max_ndim = max(t.ndim for t in ret)
        ret = tuple(
            t[(slice(None),) * t.ndim + (None,) * (max_ndim - t.ndim)]
            if t.ndim < max_ndim
            else t
            for t in ret
        )
        return ret


def get_transformed_lf(
    R_d: float = 0.3,
    T_0: float = 5.0,
    n_iter_eps: int = 5,
    n_iter_a: int = 100,
    points: int = 1000,
):
    R_ap = 0.048 * R_d - 0.01
    R_kp = 0.118 * R_d + 0.224
    R_gp = 0.25 * R_kp * (0.5 + 1.2 * R_kp) / (0.11 * R_d - R_ap * (0.5 + 1.2 * R_kp))

    # T_e = (1 + R_kp) * 0.5 / R_gp * T_0

    # T_b = T_0 - T_e

    T_a = R_ap * T_0
    T_p = 0.5 * T_0 / R_gp
    T_e = T_p * (R_kp + 1)
    T_b = T_0 - T_e

    omega_g = math.pi / T_p

    E_e = 1

    a = 1
    eps = 1

    for i in range(n_iter_eps):
        f_eps = eps * T_a + math.expm1(-eps * T_b)
        f_eps_grad = T_a - T_b * math.exp(-eps * T_b)
        eps = eps - f_eps / f_eps_grad
        eps = abs(eps)

    for i in range(n_iter_a):
        E_0 = -E_e * math.exp(-a * T_e) / math.sin(omega_g * T_e)
        A_o = E_0 * math.exp(a * T_e) / math.sqrt(omega_g**2 + a**2) * math.sin(
            omega_g * T_e - math.atan(omega_g / a)
        ) + E_0 * omega_g / (omega_g**2 + a**2)
        A_r = -E_e / (eps**2 * T_a) * (1 - math.exp(-eps * T_b) * (1 + eps * T_b))
        f_a = A_o + A_r
        f_a_grad = (1 - 2 * a * A_r / E_e) * math.sin(
            omega_g * T_e
        ) - omega_g * T_e * math.exp(-a * T_e)
        a = a - f_a / f_a_grad

    t = torch.linspace(0, T_0, points + 1)[:-1]
    before_T_e = t[t < T_e]
    after_T_e = t[t >= T_e]
    before = E_0 * torch.exp(a * before_T_e) * torch.sin(omega_g * before_T_e)
    after = (
        -E_e / eps / T_a * (torch.exp(-eps * (after_T_e - T_e)) - math.exp(-eps * T_b))
    )
    return torch.cat([before, after])


def get_radiation_time_filter(
    num_zeros: int = 16, window_fn: Callable[[int], torch.Tensor] = None
):
    t = torch.arange(-num_zeros, num_zeros + 1)
    pi_t = t * torch.pi
    tmp = torch.cos(pi_t) - torch.sinc(pi_t)
    out = tmp / t
    out[num_zeros] = 0

    if window_fn is not None:
        out *= window_fn(out.shape[0])
    return out


def get_window_fn(window: str = "hann"):
    if window == "hanning":
        return torch.hann_window
    elif window == "hamming":
        return torch.hamming_window
    elif window == "blackman":
        return torch.blackman_window
    elif window == "bartlett":
        return torch.bartlett_window
    else:
        try:
            return lambda n: torch.tensor(get_window(window, n))
        except:
            raise ValueError(f"Unknown window function {window}")


def fir_filt(x: torch.Tensor, h: torch.Tensor):
    """
    x: (batch, seq_len)
    h: (batch, seq_len, filter_len)
    """
    x = F.pad(x, (h.shape[-1] - 1, 0)).unfold(-1, h.shape[-1], 1)
    return (
        torch.matmul(x.unsqueeze(-2), h.flip(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
    )


def coeff_product(polynomials: Union[Tensor, List[Tensor]]) -> Tensor:
    n = len(polynomials)
    if n == 1:
        return polynomials[0]

    c1 = coeff_product(polynomials[n // 2 :])
    c2 = coeff_product(polynomials[: n // 2])
    if c1.shape[1] > c2.shape[1]:
        c1, c2 = c2, c1
    weight = c1.unsqueeze(1).flip(2)
    prod = F.conv1d(
        c2.unsqueeze(0),
        weight,
        padding=weight.shape[2] - 1,
        groups=c2.shape[0],
    ).squeeze(0)
    return prod


def complex2biquads(roots: Tensor) -> Tensor:
    assert roots.is_complex()
    mag = roots.abs()
    a1 = -2 * roots.real
    a2 = mag.square()
    return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)


def params2biquads(param1: Tensor, param2: Tensor) -> Tensor:
    assert torch.all(param1 >= -1) and torch.all(param1 <= 1)
    assert torch.all(param2 >= -1) and torch.all(param2 <= 1)
    a1 = 2 * param1
    a1_abs = a1.abs()
    a2 = 0.5 * ((2 - a1_abs) * param2 + a1_abs)
    return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)


def biquads2lpc(biquads: Tensor) -> Tensor:
    assert biquads.shape[-1] == 3
    return coeff_product(biquads.view(-1, *biquads.shape[-2:]).transpose(0, 1)).view(
        *biquads.shape[:-2], -1
    )[..., 1:]


def get_logits2biquads(
    rep_type: str,
    max_abs_pole: float = 0.99,
) -> Callable:
    if rep_type == "coef":

        def logits2coeff(logits: Tensor) -> Tensor:
            assert logits.shape[-1] == 2
            a1 = torch.tanh(logits[..., 0]) * max_abs_pole * 2
            a1_abs = torch.abs(a1)
            a2 = 0.5 * (
                (2 - a1_abs) * torch.tanh(logits[..., 1]) * max_abs_pole + a1_abs
            )
            return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)

    elif rep_type == "conj":

        def logits2coeff(logits: Tensor) -> Tensor:
            assert logits.shape[-1] == 2
            mag = torch.sigmoid(logits[..., 0]) * max_abs_pole
            cos = torch.tanh(logits[..., 1])
            a1 = -2 * mag * cos
            a2 = mag.square()
            return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)

    elif rep_type == "real":

        def logits2coeff(logits: Tensor) -> Tensor:
            assert logits.shape[-1] == 2
            z1 = torch.tanh(logits[..., 0]) * max_abs_pole
            z2 = torch.tanh(logits[..., 1]) * max_abs_pole
            a1 = -z1 - z2
            a2 = z1 * z2
            return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)

    else:
        raise ValueError(f"Unknown rep_type: {rep_type}, expected coef, conj or real")

    return logits2coeff


class TimeContext(object):
    hop_length: int

    def __init__(self, hop_length: int):
        self.hop_length = hop_length

    def __call__(self, hop_length: int):
        return TimeContext(hop_length * self.hop_length)


def linear_upsample(ctx: TimeContext, x: Tensor) -> Tensor:
    return F.interpolate(
        x.reshape(-1, 1, x.size(-1)),
        (x.size(-1) - 1) * ctx.hop_length + 1,
        mode="linear",
        align_corners=True,
    ).view(*x.shape[:-1], -1)


def smooth_phase_offset(phase_offset: Tensor) -> Tensor:
    # wrapp the differences into [-0.5, 0.5]
    return torch.cumsum(
        torch.cat(
            [phase_offset[:, :1], (phase_offset.diff(dim=1) + 0.5) % 1 - 0.5], dim=1
        ),
        dim=1,
    )


def hilbert(x: Tensor, dim: int = -1) -> Tensor:
    assert not x.is_complex()
    N = x.shape[dim]
    Xf = torch.fft.fft(x, dim=dim)
    h = x.new_zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [None] * x.ndim
        ind[dim] = slice(None)
        h = h[tuple(ind)]
    x = torch.fft.ifft(Xf * h, dim=dim)
    return x


def freq2cent(f0):
    return 1200 * np.log2(f0 / 440)


def rc2lpc(rc: Tensor) -> Tensor:
    assert rc.ndim == 3
    order = rc.shape[-1]
    if order == 1:
        return rc
    k_0 = rc[..., :1]
    current_lpc = torch.cat([torch.ones_like(k_0), k_0], dim=-1)

    for n in range(1, order):
        prev_lpc = torch.cat([current_lpc, torch.zeros_like(k_0)], dim=-1)
        k_n = rc[..., n : n + 1]
        current_lpc = prev_lpc + k_n * prev_lpc.flip(-1)
    return current_lpc[..., 1:]


get_f0 = partial(
    pw.dio,
    # f0_floor=65,
    f0_ceil=1047,
    channels_in_octave=2,
    frame_period=5,
)
