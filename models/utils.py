import torch
from torch import Tensor
import torch.nn.functional as F
import math
from typing import Callable, Optional, Tuple, Union, List


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
            a1 = 2 * torch.tanh(logits[..., 0]) * max_abs_pole
            a1_abs = a1.abs()
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


def linear_upsample(x: Tensor, ctx: TimeContext) -> Tensor:
    return F.interpolate(
        x.view(-1, 1, x.size(-1)),
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
