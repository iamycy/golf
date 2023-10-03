import torch
from typing import Optional, Tuple, Union, List, Callable, Any

from .synth import OscillatorInterface
from .filters import FilterInterface, LTVFilterInterface
from .noise import NoiseInterface
from .utils import AudioTensor
from .ctrl import PassThrough, Synth


class HarmonicPlusNoiseSynth(Synth):
    def __init__(
        self,
        harm_oscillator: OscillatorInterface,
        noise_generator: NoiseInterface,
        harm_filter: Union[LTVFilterInterface, PassThrough],
        noise_filter: Union[LTVFilterInterface, PassThrough],
        end_filter: Union[FilterInterface, PassThrough],
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
        harm_oscillator_params: Tuple[AudioTensor, ...],
        noise_generator_params: Tuple[AudioTensor, ...],
        harm_filter_params: Tuple[AudioTensor, ...],
        noise_filter_params: Tuple[AudioTensor, ...],
        voicing: Optional[AudioTensor] = None,
        **other_params
    ) -> AudioTensor:
        # Time-varying components
        harm_osc = self.harm_oscillator(phase, *harm_oscillator_params)
        if voicing is not None:
            assert torch.all(voicing >= 0) and torch.all(voicing <= 1)
            harm_osc = harm_osc * voicing

        noise = self.noise_generator(harm_osc, *noise_generator_params)

        harm_osc = self.harm_filter(harm_osc, *harm_filter_params)
        noise = self.noise_filter(noise, *noise_filter_params)

        out = harm_osc + noise

        # Static components
        return self.end_filter(out)
