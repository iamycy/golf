import torch
from torch import nn
from typing import Optional, Tuple

from .synth import OscillatorInterface
from .filters import LTVFilterInterface
from .noise import NoiseInterface
from .utils import TimeTensor


class SourceFilterSynth(nn.Module):
    def __init__(
        self,
        harm_oscillator: OscillatorInterface,
        noise_generator: NoiseInterface,
        noise_filter: LTVFilterInterface,
        end_filter: LTVFilterInterface,
    ):
        super().__init__()

        # Time-varying components
        self.harm_oscillator = harm_oscillator
        self.noise_generator = noise_generator
        self.noise_filter = noise_filter
        self.end_filter = end_filter

    def forward(
        self,
        phase: TimeTensor,
        harm_osc_params: Tuple[TimeTensor, ...],
        noise_params: Tuple[TimeTensor, ...],
        noise_filt_params: Tuple[TimeTensor, ...],
        end_filt_params: Tuple[TimeTensor, ...],
        voicing: Optional[TimeTensor] = None,
    ) -> TimeTensor:

        # Time-varying components
        harm_osc = self.harm_oscillator(phase, *harm_osc_params)
        if voicing is not None:
            assert torch.all(voicing >= 0) and torch.all(voicing <= 1)
            harm_osc = harm_osc * voicing

        filtered_noise = self.noise_filter(
            self.noise_generator(harm_osc, *noise_params), *noise_filt_params
        )

        minimum_length = min(harm_osc.size("T"), filtered_noise.size("T"))

        src = (
            harm_osc.truncate(minimum_length)
            + filtered_noise.truncate(minimum_length)
            - self.noise_filter(harm_osc, *noise_filt_params).truncate(minimum_length)
        )
        return self.end_filter(src, *end_filt_params)
