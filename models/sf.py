import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from .synth import OscillatorInterface
from .filters import LTVFilterInterface
from .noise import NoiseInterface
from .utils import TimeContext, linear_upsample


class SourceFilterSynth(nn.Module):
    def __init__(
        self,
        harm_oscillator: OscillatorInterface,
        noise_generator: NoiseInterface,
        noise_filter: LTVFilterInterface,
        end_filter: LTVFilterInterface,
        harm_filter: Optional[LTVFilterInterface] = None,
        use_noise_filter_on_harm: bool = False,
    ):
        super().__init__()

        # Time-varying components
        self.harm_oscillator = harm_oscillator
        self.noise_generator = noise_generator
        self.noise_filter = noise_filter
        self.harm_filter = harm_filter
        self.end_filter = end_filter
        self.use_noise_filter_on_harm = use_noise_filter_on_harm

    def forward(
        self,
        ctx: TimeContext,
        phase_params: Tuple[Tensor, Optional[Tensor]],
        harm_osc_params: Tuple[Tensor, ...],
        noise_filt_params: Tuple[Tensor, ...],
        end_filt_params: Tuple[Tensor, ...],
        harm_filt_params: Tuple[Tensor, ...] = None,
        noise_params: Tuple[Tensor, ...] = (),
    ) -> Tensor:
        """
        Args:
            phase: (batch_size, samples)
        """
        phase, *_ = phase_params
        assert torch.all(phase >= 0) and torch.all(phase <= 0.5)
        upsampled_phase = linear_upsample(phase, ctx)

        # Time-varying components
        harm_osc = self.harm_oscillator(upsampled_phase, *harm_osc_params, ctx=ctx)
        if len(_):
            voicing = _[0]
            assert torch.all(voicing >= 0) and torch.all(voicing <= 1)
            upsampled_voicing = linear_upsample(voicing, ctx)
            harm_osc = harm_osc * upsampled_voicing[:, : harm_osc.shape[1]]

        noise = self.noise_generator(harm_osc, *noise_params, ctx=ctx)

        if self.harm_filter is not None:
            harm_osc = self.harm_filter(harm_osc, *harm_filt_params, ctx=ctx)
        elif self.use_noise_filter_on_harm:
            filtered = self.noise_filter(harm_osc, *noise_filt_params, ctx=ctx)
            harm_osc = harm_osc[:, : filtered.shape[1]] + filtered

        noise = self.noise_filter(noise, *noise_filt_params, ctx=ctx)

        return self.end_filter(harm_osc + noise, *end_filt_params, ctx=ctx)
