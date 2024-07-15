import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchaudio.functional import lfilter, melscale_fbanks
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from torch_fftconv.functional import fft_conv1d
from typing import Optional, Union, List, Tuple, Callable, Any
from warnings import warn
from diffsptk import (
    MLSA,
    MelCepstralAnalysis,
    MelGeneralizedCepstrumToSpectrum,
    PQMF,
    IPQMF,
)
from diffsptk.functional import lsp2lpc
from torchlpc import sample_wise_lpc


from .audiotensor import AudioTensor


from .lpc import lpc_synthesis
from .utils import (
    get_radiation_time_filter,
    get_window_fn,
    coeff_product,
    complex2biquads,
    params2biquads,
    hilbert,
    fir_filt,
    rc2lpc,
    get_logits2biquads,
    biquads2lpc,
)
from .ctrl import Controllable, wrap_ctrl_fn

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

    def reverse(self, ex: AudioTensor, *args, **kwargs) -> AudioTensor:
        raise NotImplementedError


class LTVMinimumPhaseFilterPrecise(LTVFilterInterface):
    def __init__(
        self,
        lpc_order: int = None,
        lpc_parameterisation: str = "rc2lpc",
        max_abs_value: float = 1.0,
    ):
        super().__init__()

        if lpc_parameterisation in ("coef", "conj", "real"):
            logits2biquads = get_logits2biquads(lpc_parameterisation, max_abs_value)
            logits2lpc = lambda logits: biquads2lpc(
                logits2biquads(logits.view(logits.shape[0], logits.shape[1], -1, 2))
            )
            num_logits = lpc_order
        elif lpc_parameterisation == "rc2lpc":
            logits2lpc = lambda logits: rc2lpc(logits.tanh() * max_abs_value)
            num_logits = lpc_order
        elif lpc_parameterisation == "lsp2lpc":
            logits2lpc = lambda logits: lsp2lpc(
                logits.softmax(-1).cumsum(-1).roll(1, -1) * torch.pi
            )[..., 1:]
            num_logits = lpc_order + 1
        else:
            raise ValueError(f"Unknown lpc_parameterisation: {lpc_parameterisation}")

        if lpc_order is not None:
            self.ctrl = wrap_ctrl_fn(
                split_size=(1, num_logits),
                trsfm_fn=lambda log_gain, lpc_logits: (
                    torch.exp(log_gain),
                    lpc_logits.new_tensor(logits2lpc(lpc_logits.as_tensor())),
                ),
            )

    def forward(self, ex: AudioTensor, gain: AudioTensor, a: AudioTensor):
        assert ex.ndim == 2
        assert gain.ndim == 2
        assert a.ndim == 3
        assert a.shape[1] == gain.shape[1]
        device = ex.device
        dtype = ex.dtype

        ex = ex * gain
        ex = ex.as_tensor()
        a = a.reduce_hop_length().as_tensor()[:, : ex.shape[1]]
        ex = ex[:, : a.shape[1]]

        y = sample_wise_lpc(ex, a)
        return AudioTensor(y.to(device).to(dtype))


class LTVMinimumPhaseFilter(LTVMinimumPhaseFilterPrecise):
    def __init__(
        self,
        window: str,
        window_length: int,
        centred: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        window = get_window_fn(window)(window_length)
        self.register_buffer(
            "_kernel", torch.diag(window).unsqueeze(1), persistent=False
        )
        self.centred = centred

    def forward(self, ex: AudioTensor, gain: AudioTensor, a: AudioTensor):
        """
        Args:
            ex (Tensor): [B, T]
            gain (Tensor): [B, T / hop_length]
            a (Tensor): [B, T / hop_length, order]
            ctx (TimeContext): TimeContext
        """
        assert a.shape[1] == gain.shape[1]

        hop_length = gain.hop_length

        window_size = self._kernel.shape[0]
        assert window_size >= hop_length * 2, f"{window_size} < {hop_length * 2}"
        padding = window_size // 2 if self.centred else (window_size - hop_length) // 2

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

    def reverse(
        self, ex: AudioTensor, y: AudioTensor, gain: AudioTensor, a: AudioTensor
    ) -> Tuple[AudioTensor, AudioTensor]:
        upsampled_a = a.reduce_hop_length().as_tensor()
        fir = torch.cat([torch.ones_like(upsampled_a[..., :1]), upsampled_a], dim=-1)
        y = y[:, : fir.shape[1]]
        fir = fir[:, : y.shape[1]]
        y_ex = fir_filt(y.as_tensor(), fir)
        y_ex = AudioTensor(y_ex)
        return ex * gain, y_ex


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

        if n_mag is not None:
            self.ctrl = wrap_ctrl_fn(split_size=(n_mag,), trsfm_fn=lambda x: (x,))

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
        # assert (
        #     unfolded.shape[1] <= kernel.shape[1]
        # ), f"{unfolded.shape} != {kernel.shape}"
        kernel = kernel.as_tensor()[:, : unfolded.shape[1]]
        unfolded = unfolded[:, : kernel.shape[1]]

        convolved = self.convolve_fn(
            unfolded.reshape(1, -1, unfolded.shape[-1]),
            kernel.reshape(-1, 1, kernel.shape[-1]),
            groups=kernel.shape[0] * kernel.shape[1],
        ).view(kernel.shape[0], -1)
        return AudioTensor(convolved)


class LTVAPZeroPhaseFIRFilter(LTVZeroPhaseFIRFilter):
    def __init__(self, window: str, conv_method: str = "direct", n_mag: int = None):
        super().__init__(window, conv_method, n_mag)

        n_fft = 2 * (n_mag - 1)

        if n_mag is not None:
            self.ctrl = wrap_ctrl_fn(
                split_size=(n_mag,),
                trsfm_fn=lambda x: (torch.log(torch.sigmoid(x) * n_fft**0.5),),
            )


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
            persistent=False,
        )
        self._padding = self._kernel.size(-1) // 2

    def forward(self, ex: Tensor):
        assert ex.ndim == 2
        return F.conv1d(
            ex.unsqueeze(1),
            self._kernel,
            padding=self._padding,
        ).squeeze(1)


class LTIAcousticFilter(FilterInterface):
    def __init__(
        self,
        length: int,
        conv_method: str = "direct",
    ):
        super().__init__()
        self.kernel = nn.Parameter(torch.zeros(length - 1))
        self._padding = length - 1

        if conv_method == "direct":
            self.conv_func = F.conv1d
        elif conv_method == "fft":
            self.conv_func = fft_conv1d
        else:
            raise ValueError(f"Unknown conv_method: {conv_method}")

    def forward(self, ex: AudioTensor):
        zero_padded = F.pad(
            ex.as_tensor()[:, None, :-1], (self._padding, 0), "constant", 0
        )
        zero_padded_filtered = self.conv_func(
            zero_padded, self.kernel[None, None, :]
        ).squeeze(1)
        return ex + AudioTensor(zero_padded_filtered)

    @property
    def impulse_response(self):
        return torch.cat([self.kernel, torch.ones(1, device=self.kernel.device)]).flip(
            0
        )


class LTVPQMF(LTVFilterInterface):
    def __init__(self, n_mag: int, filter_order: int, alpha: float = 0.0):
        super().__init__()

        self.pqmf = PQMF(n_mag, filter_order, alpha=alpha)
        self.ipqmf = IPQMF(n_mag, filter_order, alpha=alpha)

        self.ctrl = wrap_ctrl_fn(
            split_size=(n_mag,),
            trsfm_fn=lambda x: (x,),
        )

    def forward(self, ex: AudioTensor, log_gain: AudioTensor):
        gain = torch.exp(log_gain)
        ex = ex.as_tensor().unsqueeze(1)
        bands = fft_conv1d(self.pqmf.pad(ex), self.pqmf.filters)
        filtered = AudioTensor(bands.mT) * gain
        # return AudioTensor(
        #     F.conv1d(
        #         self.ipqmf.pad(filtered.as_tensor().mT), self.ipqmf.filters
        #     ).squeeze(1)
        # )
        return torch.sum(filtered, dim=2)


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
        **kwargs,
    ) -> None:
        super().__init__()

        self.mlsa = MLSA(
            filter_order,
            frame_period=frame_period,
            **kwargs,
        )

        self.ctrl = wrap_ctrl_fn(
            split_size=(filter_order + 1,),
            trsfm_fn=lambda x: (x,),
        )

    def forward(self, ex: AudioTensor, mc: AudioTensor, **kwargs):
        assert mc.hop_length == self.mlsa.frame_period
        ex = ex.as_tensor()
        mc = mc.as_tensor()
        minimum_frames = ex.shape[1] // self.mlsa.frame_period
        ex = ex[:, : minimum_frames * self.mlsa.frame_period]
        mc = mc[:, :minimum_frames]
        return AudioTensor(self.mlsa(ex, mc))


class LTVCepFilter(LTVFilterInterface):
    def __init__(
        self,
        filter_order: int,
        n_fft: int,
        window: str,
        hop_length: int,
        phase: str = "zero",
        **kwargs,
    ) -> None:
        super().__init__()

        assert n_fft % 2 == 0

        self.stft = Spectrogram(
            n_fft=n_fft,
            window_fn=get_window_fn(window),
            hop_length=hop_length,
            power=None,
            center=True,
            onesided=False,
            **kwargs,
        )
        self.istft = InverseSpectrogram(
            n_fft=n_fft,
            window_fn=get_window_fn(window),
            hop_length=hop_length,
            center=True,
            onesided=False,
            **kwargs,
        )

        if phase not in ["zero", "min"]:
            raise ValueError(f"Unknown phase: {phase}")

        self.n_fft = n_fft
        self.filter_order = filter_order
        self.hop_length = hop_length
        self.phase = phase

        self.pad = nn.Sequential(
            nn.ConstantPad1d((0, n_fft // 2 - filter_order), 0),
            nn.ReflectionPad1d((0, n_fft // 2 - 1)),
        )

        self.ctrl = wrap_ctrl_fn(
            split_size=(filter_order + 1,),
            trsfm_fn=lambda x: (x,),
        )

    def forward(self, ex: AudioTensor, ceps: AudioTensor, **kwargs):
        assert ceps.hop_length == self.hop_length
        ex = ex.as_tensor()
        ceps = ceps.as_tensor()
        log_mag = torch.fft.fft(self.pad(ceps), dim=-1).real

        if self.phase == "zero":
            H = torch.exp(log_mag).transpose(-1, -2)
        else:
            min_phase = -hilbert(log_mag, dim=-1).imag
            H = torch.exp(log_mag + 1j * min_phase).transpose(-1, -2)

        X = self.stft(ex)[..., : H.shape[-1]]
        H = H[..., : X.shape[-1]]
        return AudioTensor(self.istft(X * H))


class LTVMLSAFilter2(LTVMLSAFilter):
    def __init__(
        self,
        n_fft: int,
        frame_period: int,
        filter_order: int,
        *args,
        window: str = "hanning",
        alpha: float = 0,
        gamma: float = 0,
        **kwargs,
    ):
        super().__init__(
            *args,
            filter_order=filter_order,
            frame_period=frame_period,
            alpha=alpha,
            gamma=gamma,
            **kwargs,
        )
        self.stft = Spectrogram(
            n_fft=n_fft,
            hop_length=frame_period,
            window_fn=get_window_fn(window),
            power=None,
            center=True,
            onesided=False,
        )

        self.istft = InverseSpectrogram(
            n_fft=n_fft,
            hop_length=frame_period,
            window_fn=get_window_fn(window),
            center=True,
            onesided=False,
        )

        self.mc2sp = MelGeneralizedCepstrumToSpectrum(
            filter_order,
            n_fft,
            alpha=alpha,
            gamma=gamma,
            out_format="log-magnitude",
            n_fft=n_fft,
        )

    def forward(self, ex: AudioTensor, mc: AudioTensor, **kwargs):
        assert mc.hop_length == self.mlsa.frame_period
        ex = ex.as_tensor()
        mc = mc.as_tensor()

        log_mag = self.mc2sp(mc)
        log_mag = torch.cat([log_mag, log_mag.flip(-1)[..., 1:-1]], dim=-1)
        min_phase = -hilbert(log_mag, dim=-1).imag
        H = torch.exp(log_mag + 1j * min_phase).transpose(-1, -2)

        X = self.stft(ex)[..., : H.shape[-1]]
        H = H[..., : X.shape[-1]]
        return AudioTensor(self.istft(X * H))


class LTVAPFilter(LTVMLSAFilter):
    def __init__(
        self,
        n_mag: int,
        filter_order: int,
        frame_period: int,
        *args,
        alpha: float = 0,
        gamma: float = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            filter_order,
            frame_period,
            *args,
            phase="zero",
            alpha=alpha,
            gamma=gamma,
            **kwargs,
        )

        n_fft = n_mag * 2 - 2
        self.mcep = MelCepstralAnalysis(filter_order, n_fft, alpha=alpha, gamma=gamma)

        self.ctrl = wrap_ctrl_fn(
            split_size=(n_mag,),
            trsfm_fn=lambda x: (x.new_tensor(self.mcep(torch.sigmoid(x.as_tensor()))),),
        )


class DiffWorldSPFilter(LTVFilterInterface):
    def __init__(
        self,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        f_min: float,
        f_max: float,
        center: bool = True,
        window: str = "hanning",
        **kwargs,
    ) -> None:
        super().__init__()
        fb = melscale_fbanks(n_fft // 2 + 1, f_min, f_max, n_mels, **kwargs)
        inv_fb = torch.linalg.pinv(fb).relu()
        self.register_buffer("fb", inv_fb, persistent=False)

        self.stft = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            window_fn=get_window_fn(window),
            power=None,
            center=center,
        )
        self.istft = InverseSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            window_fn=get_window_fn(window),
            center=center,
        )

        self.ctrl = wrap_ctrl_fn(
            split_size=(n_mels,),
            trsfm_fn=lambda x: (torch.exp(x),),
        )

    def forward(self, ex: AudioTensor, mel_sp: AudioTensor):
        assert mel_sp.hop_length == self.stft.hop_length
        sp = mel_sp @ self.fb
        sp = torch.transpose(torch.sqrt(sp), 1, 2).as_tensor()
        X = self.stft(ex.as_tensor())
        X = X[..., : sp.shape[-1]]
        sp = sp[..., : X.shape[-1]]
        return AudioTensor(self.istft(X * sp))


class SampleBasedLTVMinimumPhaseFilter(LTVMinimumPhaseFilter):
    def __init__(
        self,
        lpc_order: int = None,
        lpc_parameterisation: str = "rc2lpc",
        max_abs_value: float = 1,
        **kwargs,
    ):
        warn(
            "SampleBasedLTVMinimumPhaseFilter is deprecated. Use LTVMinimumPhaseFilterPrecise instead."
        )
        super().__init__("hanning", 1, lpc_order, lpc_parameterisation, max_abs_value)

    def forward(self, ex: AudioTensor, gain: AudioTensor, a: AudioTensor):
        assert ex.ndim == 2
        assert gain.ndim == 2
        assert a.ndim == 3
        assert a.shape[1] == gain.shape[1]
        device = ex.device
        dtype = ex.dtype

        ex = ex * gain
        ex = ex.as_tensor()
        a = a.reduce_hop_length().as_tensor()[:, : ex.shape[1]]
        ex = ex[:, : a.shape[1]]

        y = sample_wise_lpc(ex, a)
        return AudioTensor(y.to(device).to(dtype))


def convert2samplewise(config: dict):
    for key, value in config.items():
        if key == "class_path":
            if ".LTVMinimumPhaseFilter" in config["class_path"]:
                config["class_path"] = "models.filters.LTVMinimumPhaseFilterPrecise"
                return config
            elif ".LTVMinimumPhaseFIRFilter" in config["class_path"]:
                config["class_path"] = "models.filters.LTVMinimumPhaseFIRFilterPrecise"
                config["init_args"].pop("conv_method")
                return config
            elif ".LTVZeroPhaseFIRFilter" in config["class_path"]:
                config["class_path"] = "models.filters.LTVZeroPhaseFIRFilterPrecise"
                config["init_args"].pop("conv_method")
                return config
        elif isinstance(value, dict):
            config[key] = convert2samplewise(value)
    return config
