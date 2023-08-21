import torch
from torch import nn, Tensor
from typing import Optional, Union, List, Tuple, Callable
import math
from scipy import signal

from .ctrl import Controllable, SPLIT_TRSFM_SIGNATURE, TRSFM_TYPE
from .utils import AudioTensor

__all__ = [
    "NoiseInterface",
    "StandardNormalNoise",
    "UniformNoise",
    "SignFlipNoise",
    "NoiseBand",
]


class NoiseInterface(Controllable):
    dist: torch.distributions.Distribution

    def __init__(self, dist: torch.distributions.Distribution):
        super().__init__()
        self.dist = dist

    def forward(self, ref: AudioTensor, *args, **kwargs) -> AudioTensor:
        return self.dist.sample(ref.shape).to(ref.device)


class StandardNormalNoise(NoiseInterface):
    def __init__(self):
        super().__init__(torch.distributions.Normal(0, 1))

    def forward(self, ref: AudioTensor, *args, **kwargs) -> AudioTensor:
        return torch.randn_like(ref)


class UniformNoise(NoiseInterface):
    def __init__(self):
        super().__init__(torch.distributions.Uniform(-math.sqrt(3), math.sqrt(3)))

    def forward(self, ref: AudioTensor, *args, **kwargs) -> AudioTensor:
        return (torch.rand_like(ref) - 0.5) * 2 * math.sqrt(3)


class SignFlipNoise(NoiseInterface):
    def __init__(self):
        super().__init__(torch.distributions.Uniform(-1, 1))

    def forward(self, ref: AudioTensor, *args, **kwargs) -> AudioTensor:
        sign = ref.as_tensor().new_empty(ref.shape[:-1]).uniform_(-1, 1).sign()
        tmp = torch.ones_like(ref).as_tensor()
        tmp[..., ::2] = sign.unsqueeze(-1)
        tmp[..., 1::2] = -sign.unsqueeze(-1)
        return ref.new_tensor(tmp)


class NoiseBand(NoiseInterface):
    """Filterbank class that builds a filterbank with linearly and logarithmically distributed filters.

    Args:
    ----------
    n_filters : int
        Number of linearly distributed filters
    fs : int
        Sampling rate
    attenuation : float
        FIR filter attenuation used in the Kaiser window (in dB)
    """

    def __init__(
        self,
        n_filters=1024,
        fs=44100,
        attenuation=50,
        normalize_noise_bands=True,
    ):
        super().__init__(torch.distributions.Normal(0, 1))
        print("Building filterbank...")
        frequency_bands = self.get_linear_bands(
            n_filters_linear=n_filters,
            fs=fs,
        )
        band_centers = self.get_band_centers(frequency_bands=frequency_bands, fs=fs)
        filters = self.build_filterbank(
            frequency_bands=frequency_bands, fs=fs, attenuation=attenuation
        )
        # self.register_buffer("filters", filters)
        self.register_buffer("band_centers", band_centers.float())

        max_filter_len = max(len(array) for array in filters)
        noise_bands = self.get_noise_bands(
            filters=filters,
            min_noise_len=max_filter_len,
            normalize=normalize_noise_bands,
        )
        self.register_buffer("noise_bands", noise_bands.float())

        print(f"Done. {len(filters)} filters, max filter length: {max_filter_len}")

        def ctrl_fn(other_split_trsfm: SPLIT_TRSFM_SIGNATURE):
            def split_and_trsfm(
                split_sizes: Tuple[Tuple[int, ...], ...],
                trsfm_fns: Tuple[TRSFM_TYPE, ...],
            ):
                split_sizes = split_sizes + ((n_filters,),)
                trsfm_fns = trsfm_fns + (lambda x: (x,),)
                return other_split_trsfm(split_sizes, trsfm_fns)

            return split_and_trsfm

        self.ctrl = ctrl_fn

    def forward(self, ref: AudioTensor, log_gain: AudioTensor) -> AudioTensor:
        gain = torch.exp(log_gain)
        B, T = ref.shape
        num_bands, bands_len = self.noise_bands.shape
        rand_offset = torch.randint(0, bands_len, (B, num_bands)).to(ref.device)
        index = torch.arange(T).to(ref.device)
        wrapped_index = (index + rand_offset[:, :, None]) % bands_len
        band_offset = torch.arange(num_bands).to(ref.device) * bands_len
        wrapped_index += band_offset[:, None]
        noise = self.noise_bands.view(-1)[wrapped_index.view(-1)].view(B, num_bands, T)
        return torch.sum(AudioTensor(noise.mT) * gain, dim=2)

    def get_linear_bands(self, n_filters_linear, fs):
        linear_bands = torch.linspace(0, fs / 2, n_filters_linear + 1)
        linear_bands = torch.vstack((linear_bands[1:-2], linear_bands[2:-1])).T
        return linear_bands

    def get_band_centers(self, frequency_bands, fs):
        mean_frequencies = torch.mean(frequency_bands, axis=1)
        lower_edge = frequency_bands[0, 0] / 2
        upper_edge = ((fs / 2) + frequency_bands[-1, -1]) / 2
        out = mean_frequencies.new_empty((mean_frequencies.shape[0] + 2,))
        out[0] = lower_edge
        out[1:-1] = mean_frequencies
        out[-1] = upper_edge
        return out

    def get_filter(
        self, cutoff, fs, attenuation, pass_zero, transition_bandwidth=0.2, scale=True
    ):
        if cutoff.numel() > 1:  # BPF
            bandwidth = abs(cutoff[1] - cutoff[0])
        elif pass_zero == True:  # LPF
            bandwidth = cutoff
        elif pass_zero == False:  # HPF
            bandwidth = abs((fs / 2) - cutoff)
        width = (bandwidth / (fs / 2)) * transition_bandwidth
        N, beta = signal.kaiserord(ripple=attenuation, width=width)
        N = 2 * (N // 2) + 1  # make odd
        h = signal.firwin(
            numtaps=N,
            cutoff=cutoff,
            window=("kaiser", beta),
            scale=scale,
            fs=fs,
            pass_zero=pass_zero,
        )
        return torch.from_numpy(h)

    def build_filterbank(self, frequency_bands, fs, attenuation):
        filters = []
        for i in range(frequency_bands.shape[0]):
            # low pass filter
            if i == 0:
                h = self.get_filter(
                    cutoff=frequency_bands[i, 0],
                    fs=fs,
                    attenuation=attenuation,
                    pass_zero=True,
                )
                filters.append(h)
            # band pass filter
            h = self.get_filter(
                cutoff=frequency_bands[i],
                fs=fs,
                attenuation=attenuation,
                pass_zero=False,
            )
            filters.append(h)
            # high pass filter
            if i == frequency_bands.shape[0] - 1:
                h = self.get_filter(
                    cutoff=frequency_bands[i, -1],
                    fs=fs,
                    attenuation=attenuation,
                    pass_zero=False,
                )
                filters.append(h)
        return filters

    def get_noise_bands(self, filters, min_noise_len, normalize):
        # build deterministic loopable noise bands

        noise_len = 2 ** math.ceil(math.log2(min_noise_len))

        for i in range(len(filters)):
            filters[i] = torch.cat(
                [torch.zeros(noise_len - len(filters[i])), filters[i]]
            )
        filters = torch.stack(filters)
        magnitude_filters = torch.fft.rfft(filters).abs()
        phase_noise = torch.rand_like(magnitude_filters) * 2 * torch.pi
        phase_noise = torch.exp(1j * phase_noise)
        phase_noise[:, 0] = 0
        phase_noise[:, -1] = 0
        magphase = magnitude_filters * phase_noise
        noise_bands = torch.fft.irfft(magphase)
        if normalize:
            noise_bands = noise_bands / torch.max(noise_bands.abs())
        return noise_bands
