import torch
from torch import nn
from typing import Optional, Tuple

from .synth import OscillatorInterface
from .filters import LTVFilterInterface
from .noise import NoiseInterface
from .utils import AudioTensor
from .ctrl import SPLIT_TRSFM_SIGNATURE, DUMMY_SPLIT_TRSFM


class SourceFilterSynth(nn.Module):
    def __init__(
        self,
        harm_oscillator: OscillatorInterface,
        noise_generator: NoiseInterface,
        noise_filter: LTVFilterInterface,
        end_filter: LTVFilterInterface,
        subtract_harmonics: bool = True,
    ):
        super().__init__()
        self.subtract_harmonics = subtract_harmonics

        # Time-varying components
        self.harm_oscillator = harm_oscillator
        self.noise_generator = noise_generator
        self.noise_filter = noise_filter
        self.end_filter = end_filter

    def forward(
        self,
        phase: AudioTensor,
        harm_osc_params: Tuple[AudioTensor, ...],
        noise_params: Tuple[AudioTensor, ...],
        noise_filt_params: Tuple[AudioTensor, ...],
        end_filt_params: Tuple[AudioTensor, ...],
        voicing: Optional[AudioTensor] = None,
    ) -> AudioTensor:

        # Time-varying components
        harm_osc = self.harm_oscillator(phase, *harm_osc_params)
        if voicing is not None:
            assert torch.all(voicing >= 0) and torch.all(voicing <= 1)
            harm_osc = harm_osc * voicing

        src = harm_osc + self.noise_filter(
            self.noise_generator(harm_osc, *noise_params), *noise_filt_params
        )

        if self.subtract_harmonics:
            src = src - self.noise_filter(harm_osc, *noise_filt_params)

        return self.end_filter(src, *end_filt_params)

    def get_split_sizes_and_trsfms(self):
        ctrl_fns = [
            self.harm_oscillator.ctrl,
            self.noise_generator.ctrl,
            self.noise_filter.ctrl,
            self.end_filter.ctrl,
        ]
        split_trsfm = DUMMY_SPLIT_TRSFM
        for ctrl_fn in ctrl_fns[::-1]:
            split_trsfm = ctrl_fn(split_trsfm)
        return split_trsfm((), ())
