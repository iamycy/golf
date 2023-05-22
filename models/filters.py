import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchaudio.functional import lfilter
from torch_fftconv.functional import fft_conv1d
from typing import Optional, Union, List, Tuple, Callable, Any
from diffsptk import MLSA


from .lpc import lpc_synthesis
from .utils import (
    get_radiation_time_filter,
    get_window_fn,
    coeff_product,
    complex2biquads,
    params2biquads,
    AudioTensor,
    hilbert,
    fir_filt,
    rc2lpc,
    get_logits2biquads,
    biquads2lpc,
)
from .ctrl import Controllable, TRSFM_TYPE, SPLIT_TRSFM_SIGNATURE

__all__ = [
    "FilterInterface",
    "LTVMinimumPhaseFilter",
    "LTIRadiationFilter",
    "LTIComplexConjAllpassFilter",
    "LTIRealCoeffAllpassFilter",
    "LTVMinimumPhaseFIRFilterPrecise",
    "LTVMinimumPhaseFIRFilter",
    "LTVZeroPhaseFIRFilterPrecise",
    "LTVZeroPhaseFIRFilter",
]


class FilterInterface(Controllable):
    def forward(self, ex: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError


class LTVFilterInterface(FilterInterface):
    def forward(self, ex: AudioTensor, *args, **kwargs) -> AudioTensor:
        raise NotImplementedError


class LTVMinimumPhaseFilter(LTVFilterInterface):
    def __init__(
        self,
        window: str,
        window_length: int,
        lpc_order: int = None,
        lpc_parameterisation: str = "rc2lpc",
        max_abs_value: float = 1.0,
    ):
        super().__init__()
        window = get_window_fn(window)(window_length)
        self.register_buffer("_kernel", torch.diag(window).unsqueeze(1))

        def ctrl_fn(other_split_trsfm: SPLIT_TRSFM_SIGNATURE):
            if lpc_parameterisation != "rc2lpc":
                logits2biquads = get_logits2biquads(lpc_parameterisation, max_abs_value)
                logits2lpc = lambda logits: biquads2lpc(
                    logits2biquads(logits.view(logits.shape[0], logits.shape[1], -1, 2))
                )
            else:
                logits2lpc = lambda logits: rc2lpc(logits.tanh() * max_abs_value)

            def split_and_trsfm(
                split_sizes: Tuple[Tuple[int, ...], ...],
                trsfm_fns: Tuple[TRSFM_TYPE, ...],
            ):
                split_sizes = split_sizes + ((1, lpc_order),)

                def trsfm_fn(log_gain: AudioTensor, lpc_logits: AudioTensor):
                    gain = torch.exp(log_gain)
                    a = lpc_logits.new_tensor(logits2lpc(lpc_logits.as_tensor()))
                    return gain, a

                trsfm_fns = trsfm_fns + (trsfm_fn,)
                return other_split_trsfm(split_sizes, trsfm_fns)

            return split_and_trsfm

        if lpc_order is not None:
            self.ctrl = ctrl_fn

    def forward(self, ex: AudioTensor, gain: AudioTensor, a: AudioTensor):
        """
        Args:
            ex (Tensor): [B, T]
            gain (Tensor): [B, T / hop_length]
            a (Tensor): [B, T / hop_length, order]
            ctx (TimeContext): TimeContext
        """
        # ex = ex.align_to("B", "T")
        # gain = gain.align_to("B", "T")
        # a = a.align_to("B", "T", "D")
        assert a.shape[1] == gain.shape[1]

        hop_length = gain.hop_length

        window_size = self._kernel.shape[0]
        assert window_size >= hop_length * 2, f"{window_size} < {hop_length * 2}"
        padding = window_size // 2

        # interpolate gain
        # upsampled_gain = F.interpolate(
        #     gain.unsqueeze(1),
        #     scale_factor=hop_length,
        #     mode="linear",
        #     align_corners=False,
        # ).squeeze(1)
        ex = ex * gain
        ex = F.pad(
            ex,
            (padding,) * 2,
            "constant",
            0,
        )
        unfolded = ex.unfold(window_size, hop_length).as_tensor()
        a = a.as_tensor()
        gain = gain.as_tensor()
        assert unfolded.shape[1] <= a.shape[1], f"{unfolded.shape} != {a.shape}"
        a = a[:, : unfolded.shape[1]]
        gain = gain[:, : unfolded.shape[1]]

        batch, frames = gain.shape
        unfolded = unfolded.reshape(-1, window_size)
        gain = gain.reshape(-1)
        a = a.reshape(-1, a.shape[-1])
        filtered = lpc_synthesis(unfolded, torch.ones_like(gain), a).view(
            batch, frames, -1
        )

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
        return AudioTensor(y / norm)


class LTVMinimumPhaseFIRFilterPrecise(LTVFilterInterface):
    def __init__(self, window: str):
        super().__init__()
        self.window_fn = get_window_fn(window)

    @staticmethod
    def get_minimum_phase_fir(log_mag: Tensor):
        # first, get symmetric log-magnitude
        # always assume n_fft is even
        log_mag = torch.cat([log_mag, log_mag.flip(-1)[..., 1:-1]], dim=-1)
        # get minimum-phase impulse response
        min_phase = -hilbert(log_mag, dim=-1).imag
        # get minimum-phase FIR filter
        frequency_response = torch.exp(log_mag + 1j * min_phase)
        # get time-domain filter
        kernel = torch.fft.ifft(frequency_response, dim=-1).real
        return kernel

    def windowing(self, kernel: Tensor):
        window = self.window_fn(
            kernel.shape[-1], device=kernel.device, dtype=kernel.dtype
        )
        window[: kernel.shape[-1] // 2] = 1
        return kernel * window

    def forward(self, ex: AudioTensor, log_mag: AudioTensor):
        """
        Args:
            ex (Tensor): [B, T]
            log_mag (Tensor): [B, T / hop_length, n_fft // 2 + 1]
            ctx (TimeContext): TimeContext
        """
        # ex = ex.align_to("B", "T")
        # log_mag = log_mag.align_to("B", "T", "D")

        kernel = self.get_minimum_phase_fir(log_mag)
        kernel = self.windowing(kernel)

        # upsample kernel
        upsampled_kernel = log_mag.new_tensor(kernel).reduce_hop_length()

        # ex = ex[:, : upsampled_kernel.shape[1]]
        # upsampled_kernel = upsampled_kernel[:, : ex.shape[1]]
        return fir_filt(ex, upsampled_kernel)


class LTVMinimumPhaseFIRFilter(LTVMinimumPhaseFIRFilterPrecise):
    def __init__(self, window: str, conv_method: str = "direct"):
        super().__init__(window=window)
        if conv_method == "direct":
            self.convolve_fn = F.conv1d
        elif conv_method == "fft":
            self.convolve_fn = fft_conv1d
        else:
            raise ValueError(f"Unknown conv_method: {conv_method}")

    def forward(self, ex: AudioTensor, log_mag: AudioTensor):
        """
        Args:
            ex (Tensor): [B, T]
            log_mag (Tensor): [B, T / hop_length, n_fft // 2 + 1]
            ctx (TimeContext): TimeContext
        """
        # ex = ex.align_to("B", "T")
        # log_mag = log_mag.align_to("B", "T", "D")

        hop_length = log_mag.hop_length

        kernel = self.get_minimum_phase_fir(log_mag)
        kernel = self.windowing(kernel)

        # convolve
        unfolded = F.pad(ex, (kernel.shape[-1] - 1, 0), "constant", 0).unfold(
            1, kernel.shape[-1] + hop_length - 1, hop_length
        )
        assert (
            unfolded.shape[1] <= kernel.shape[1]
        ), f"{unfolded.shape} != {kernel.shape}"
        kernel = kernel[:, : unfolded.shape[1]]

        convolved = self.convolve_fn(
            unfolded.reshape(1, -1, unfolded.shape[-1]),
            kernel.reshape(-1, 1, kernel.shape[-1]).flip(-1),
            groups=kernel.shape[0] * kernel.shape[1],
        ).view(kernel.shape[0], -1)
        return convolved


class LTVZeroPhaseFIRFilterPrecise(LTVFilterInterface):
    def __init__(self, window: str, n_mag: int = None):
        super().__init__()
        self.window_fn = get_window_fn(window)

        def ctrl_fn(other_split_trsfm: SPLIT_TRSFM_SIGNATURE):
            def split_and_trsfm(
                split_sizes: Tuple[Tuple[int, ...], ...],
                trsfm_fns: Tuple[TRSFM_TYPE, ...],
            ):
                split_sizes = split_sizes + ((n_mag,),)
                trsfm_fns = trsfm_fns + (lambda x: (x,),)
                return other_split_trsfm(split_sizes, trsfm_fns)

            return split_and_trsfm

        if n_mag is not None:
            self.ctrl = ctrl_fn

    @staticmethod
    def get_zero_phase_fir(log_mag: Tensor):
        mag = torch.exp(log_mag) + 0j
        # get zero-phase impulse response
        fir = torch.fft.irfft(mag, dim=-1)
        fir = torch.fft.fftshift(fir, dim=-1)
        return fir

    def windowing(self, kernel: Tensor):
        window = self.window_fn(
            kernel.shape[-1], device=kernel.device, dtype=kernel.dtype
        )
        return kernel * window

    def forward(self, ex: AudioTensor, log_mag: AudioTensor):
        """
        Args:
            ex (Tensor): [B, T]
            log_mag (Tensor): [B, T / hop_length, n_fft // 2 + 1]
            ctx (TimeContext): TimeContext
        """
        # ex = ex.align_to("B", "T")
        # log_mag = log_mag.align_to("B", "T", "D")

        kernel = self.get_zero_phase_fir(log_mag)
        kernel = self.windowing(kernel)

        # upsampled_kernel = linear_upsample(
        #     kernel.transpose(1, 2).contiguous(), ctx
        # ).transpose(1, 2)
        upsampled_kernel = kernel.reduce_hop_length()

        # ex = ex[:, : upsampled_kernel.shape[1]]
        # upsampled_kernel = upsampled_kernel[:, : ex.shape[1]]

        padding_left = (kernel.shape[-1] - 1) // 2
        padding_right = kernel.shape[-1] - 1 - padding_left

        ex = F.pad(ex, (padding_left, padding_right), "constant", 0).unfold(
            kernel.shape[-1], 1
        )
        return torch.matmul(
            torch.unsqueeze(ex, -2), torch.unsqueeze(upsampled_kernel, -1)
        )[..., 0, 0]


class LTVZeroPhaseFIRFilter(LTVZeroPhaseFIRFilterPrecise):
    def __init__(self, window: str, conv_method: str = "direct", n_mag: int = None):
        super().__init__(window=window, n_mag=n_mag)
        if conv_method == "direct":
            self.convolve_fn = F.conv1d
        elif conv_method == "fft":
            self.convolve_fn = fft_conv1d
        else:
            raise ValueError(f"Unknown conv_method: {conv_method}")

    def forward(self, ex: AudioTensor, log_mag: AudioTensor):
        """
        Args:
            ex (Tensor): [B, T]
            log_mag (Tensor): [B, T / hop_length, n_fft // 2 + 1]
            ctx (TimeContext): TimeContext
        """
        # ex = ex.align_to("B", "T")
        # log_mag = log_mag.align_to("B", "T", "D")

        hop_length = log_mag.hop_length

        kernel = self.get_zero_phase_fir(log_mag)
        kernel = self.windowing(kernel)

        padding = (kernel.shape[-1] - 1) // 2

        # convolve
        unfolded = (
            F.pad(ex, (padding, padding), "constant", 0)
            .unfold(kernel.shape[-1] + hop_length - 1, hop_length)
            .as_tensor()
        )
        assert (
            unfolded.shape[1] <= kernel.shape[1]
        ), f"{unfolded.shape} != {kernel.shape}"
        kernel = kernel.as_tensor()[:, : unfolded.shape[1]]

        convolved = self.convolve_fn(
            unfolded.reshape(1, -1, unfolded.shape[-1]),
            kernel.reshape(-1, 1, kernel.shape[-1]),
            groups=kernel.shape[0] * kernel.shape[1],
        ).view(kernel.shape[0], -1)
        return AudioTensor(convolved)


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


class LTVMLSAFilter(LTVFilterInterface):
    def __init__(
        self,
        filter_order: int,
        frame_period: int,
        *args,
        use_ratio: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.mlsa = MLSA(
            filter_order,
            *args,
            frame_period=frame_period,
            cascade=True,
            **kwargs,
        )

        def ctrl_fn(other_split_trsfm: SPLIT_TRSFM_SIGNATURE):
            def split_and_trsfm(
                split_sizes: Tuple[Tuple[int, ...], ...],
                trsfm_fns: Tuple[TRSFM_TYPE, ...],
            ):
                split_sizes = split_sizes + ((filter_order + 1,),)
                trsfm_fns = trsfm_fns + (
                    lambda x: (torch.sigmoid(x),) if use_ratio else (x,),
                )
                return other_split_trsfm(split_sizes, trsfm_fns)

            return split_and_trsfm

        self.ctrl = ctrl_fn

    def forward(self, ex: AudioTensor, mc: AudioTensor, **kwargs):
        assert mc.hop_length == self.mlsa.frame_period
        ex = ex.as_tensor()
        mc = mc.as_tensor()
        minimum_frames = ex.shape[1] // self.mlsa.frame_period
        ex = ex[:, : minimum_frames * self.mlsa.frame_period]
        mc = mc[:, :minimum_frames]
        return AudioTensor(self.mlsa(ex, mc))
