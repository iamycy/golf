import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from .synth import OscillatorInterface
from .filters import LTVFilterInterface, FilterInterface
from .noise import NoiseInterface
from .audiotensor import AudioTensor
from .ctrl import PassThrough, Synth


class SourceFilterSynth(Synth):
    def __init__(
        self,
        harm_oscillator: OscillatorInterface,
        noise_generator: NoiseInterface,
        noise_filter: Union[LTVFilterInterface, PassThrough],
        end_filter: Union[LTVFilterInterface, PassThrough],
        room_filter: Union[FilterInterface, PassThrough] = None,
        subtract_harmonics: bool = True,
    ):
        super().__init__()
        self.subtract_harmonics = subtract_harmonics

        # Time-varying components
        self.harm_oscillator = harm_oscillator
        self.noise_generator = noise_generator
        self.noise_filter = noise_filter
        self.end_filter = end_filter

        # Room filter
        self.room_filter = room_filter if room_filter is not None else PassThrough()

    def forward(
        self,
        phase: AudioTensor,
        harm_oscillator_params: Tuple[AudioTensor, ...],
        noise_generator_params: Tuple[AudioTensor, ...],
        noise_filter_params: Tuple[AudioTensor, ...],
        end_filter_params: Tuple[AudioTensor, ...],
        voicing: Optional[AudioTensor] = None,
        target: Optional[AudioTensor] = None,
        **other_params
    ) -> AudioTensor:
        # Time-varying components
        harm_osc = self.harm_oscillator(phase, *harm_oscillator_params)
        if voicing is not None:
            assert torch.all(voicing >= 0) and torch.all(voicing <= 1)
            voicing = F.threshold(voicing, 0.5, 0)
            harm_osc = harm_osc * voicing

        src = harm_osc + self.noise_filter(
            self.noise_generator(harm_osc, *noise_generator_params),
            *noise_filter_params
        )

        if self.subtract_harmonics:
            src = src - self.noise_filter(harm_osc, *noise_filter_params)

        if target is not None:
            src, target_src = self.end_filter.reverse(src, target, *end_filter_params)
            return src, target_src
        return self.room_filter(self.end_filter(src, *end_filter_params))
