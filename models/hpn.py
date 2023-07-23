import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from .synth import OscillatorInterface
from .filters import FilterInterface, LTVFilterInterface
from .noise import NoiseInterface
from .utils import AudioTensor
from .ctrl import DUMMY_SPLIT_TRSFM


class HarmonicPlusNoiseSynth(nn.Module):
    def __init__(
        self,
        harm_oscillator: OscillatorInterface,
        noise_generator: NoiseInterface,
        harm_filter: Optional[LTVFilterInterface] = None,
        noise_filter: Optional[LTVFilterInterface] = None,
        end_filter: Optional[FilterInterface] = None,
    ):
        super().__init__()

        # Time-varying components
        self.harm_oscillator = harm_oscillator
        self.noise_generator = noise_generator
        self.harm_filter = harm_filter
        self.noise_filter = noise_filter

        # Static components
        self.end_filter = end_filter

    def forward(
        self,
        phase: AudioTensor,
        harm_osc_params: Tuple[AudioTensor, ...],
        noise_params: Tuple[AudioTensor, ...],
        harm_filt_params: Tuple[AudioTensor, ...],
        noise_filt_params: Tuple[AudioTensor, ...],
        voicing: Optional[AudioTensor] = None,
    ) -> Tensor:
        # Time-varying components
        harm_osc = self.harm_oscillator(phase, *harm_osc_params)
        if voicing is not None:
            assert torch.all(voicing >= 0) and torch.all(voicing <= 1)
            harm_osc = harm_osc * voicing

        noise = self.noise_generator(harm_osc, *noise_params)

        if self.harm_filter is not None:
            harm_osc = self.harm_filter(harm_osc, *harm_filt_params)
        if self.noise_filter is not None:
            noise = self.noise_filter(noise, *noise_filt_params)

        out = harm_osc + noise

        # Static components
        if self.end_filter is not None:
            return self.end_filter(out)
        else:
            return out

    def get_split_sizes_and_trsfms(self):
        ctrl_fns = [
            self.harm_oscillator.ctrl,
            self.noise_generator.ctrl,
            self.harm_filter.ctrl,
            self.noise_filter.ctrl,
        ]
        split_trsfm = DUMMY_SPLIT_TRSFM
        for ctrl_fn in ctrl_fns[::-1]:
            split_trsfm = ctrl_fn(split_trsfm)
        return split_trsfm((), ())
