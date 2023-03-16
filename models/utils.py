import torch
import torch.nn.functional as F
import math
from typing import Callable


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
