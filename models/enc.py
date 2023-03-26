import torch
from torch import nn, Tensor
import math
from importlib import import_module
from typing import Optional, Union, List, Tuple, Callable, Any

from .utils import get_logits2biquads, biquads2lpc, TimeContext


class BackboneModelInterface(nn.Module):
    # out_channels: int
    # last_layer_channels: int = None
    out_linear: nn.Linear

    def __init__(self, linear_in_channels: int, linear_out_channels: int):
        super().__init__()
        self.out_linear = nn.Linear(linear_in_channels, linear_out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_linear(x)


class VocoderParameterEncoderInterface(nn.Module):
    split_size: List[int]
    log_f0_min: float
    log_f0_max: float
    backbone: BackboneModelInterface

    def __init__(
        self,
        backbone_type: str,
        learn_voicing: bool = False,
        f0_min: float = 80,
        f0_max: float = 1000,
        extra_split_sizes: List[int] = [],
        **kwargs,
    ):
        super().__init__()
        extra_split_sizes = ([1, 1] if learn_voicing else [1]) + extra_split_sizes
        self.split_size = extra_split_sizes
        self.learn_voicing = learn_voicing
        self.log_f0_min = math.log(f0_min)
        self.log_f0_max = math.log(f0_max)

        module_path, class_name = backbone_type.rsplit(".", 1)
        module = import_module(module_path)

        self.backbone = getattr(module, class_name)(
            out_channels=sum(self.split_size), **kwargs
        )

    def logits2f0(self, logits: Tensor) -> Tensor:
        return torch.exp(
            logits.sigmoid() * (self.log_f0_max - self.log_f0_min) + self.log_f0_min
        )

    def forward(
        self, h: Tensor
    ) -> Tuple[Tuple[Tensor, Optional[Tensor]], ...,]:
        """
        Args:
            h: (batch_size, frames, features)
        Returns:
            harm_osc_params: Tuple[Tensor, ...]
            harm_filt_params: Tuple[Tensor, ...]
            noise_filt_params: Tuple[Tensor, ...]
            noise_params: Tuple[Tensor, ...]
        """
        f0_logits, *_ = self.backbone(h).split(self.split_size, dim=-1)
        f0 = self.logits2f0(f0_logits).squeeze(-1)
        if self.learn_voicing:
            _ = (_[0].squeeze(-1), *_[1:])
        return (f0,) + tuple(_)


class GlottalComplexConjLPCEncoder(VocoderParameterEncoderInterface):
    def __init__(
        self,
        voice_lpc_order: int,
        noise_lpc_order: int,
        table_weight_hidden_size: int,
        *args,
        max_abs_value: float = 0.99,
        use_snr: bool = False,
        extra_split_sizes: List[int] = [],
        kwargs: dict = {},
    ):
        assert voice_lpc_order % 2 == 0
        assert noise_lpc_order % 2 == 0

        extra_split_sizes.extend(
            [
                voice_lpc_order,
                1,
                noise_lpc_order,
                1,
                table_weight_hidden_size,
            ]
        )
        super().__init__(
            *args,
            extra_split_sizes=extra_split_sizes,
            **kwargs,
        )
        self.use_snr = use_snr
        self.logits2biquads = get_logits2biquads("conj", max_abs_value)
        self.backbone.out_linear.weight.data.zero_()

        offset = 1 if self.learn_voicing else 0
        self.backbone.out_linear.bias.data[
            offset + 1 : offset + 1 + voice_lpc_order : 2
        ] = -10  # initialize magnitude close to zero
        self.backbone.out_linear.bias.data[
            offset + 1 + voice_lpc_order
        ] = 0  # initialize gain
        self.backbone.out_linear.bias.data[
            offset
            + 1
            + voice_lpc_order
            + 1 : offset
            + 1
            + voice_lpc_order
            + 1
            + noise_lpc_order : 2
        ] = -10  # initialize magnitude close to zero
        if use_snr:
            self.backbone.out_linear.bias.data[
                offset + 1 + voice_lpc_order + 1 + noise_lpc_order
            ] = 20
        else:
            self.backbone.out_linear.bias.data[
                offset + 2 + voice_lpc_order + noise_lpc_order
            ] = -10

    def forward(
        self, h: Tensor
    ) -> Tuple[
        Tuple[Tensor, Optional[Tensor]],
        Tuple[Any, ...],
        Tuple[Any, ...],
        Tuple[Any, ...],
        Tuple[Any, ...],
    ]:
        batch, frames, _ = h.shape
        (
            *f0_params,
            voice_lpc_logits,
            voice_log_gain,
            noise_lpc_logits,
            noise_log_gain,
            h,
        ) = super().forward(h)
        voice_biquads = self.logits2biquads(voice_lpc_logits.view(batch, frames, -1, 2))
        noise_biquads = self.logits2biquads(noise_lpc_logits.view(batch, frames, -1, 2))
        voice_lpc_coeffs = biquads2lpc(voice_biquads)
        noise_lpc_coeffs = biquads2lpc(noise_biquads)

        if self.use_snr:
            log_snr = noise_log_gain
            noise_log_gain = voice_log_gain - log_snr * 0.5

        voice_gain = voice_log_gain.squeeze(-1).exp()
        noise_gain = noise_log_gain.squeeze(-1).exp()

        return (
            f0_params,
            (h,),
            (voice_gain, voice_lpc_coeffs),
            (noise_gain, noise_lpc_coeffs),
            (),
        )


class GlottalRealCoeffLPCEncoder(GlottalComplexConjLPCEncoder):
    def __init__(
        self,
        voice_lpc_order: int,
        noise_lpc_order: int,
        *args,
        max_abs_value: float = 0.99,
        **kwargs,
    ):
        super().__init__(
            voice_lpc_order,
            noise_lpc_order,
            *args,
            max_abs_value=max_abs_value,
            **kwargs,
        )
        self.logits2biquads = get_logits2biquads("coef", max_abs_value)
        offset = 1 if self.learn_voicing else 0
        self.backbone.out_linear.bias.data[
            offset + 1 : offset + 1 + voice_lpc_order :
        ] = 0
        self.backbone.out_linear.bias.data[
            offset
            + 1
            + voice_lpc_order
            + 1 : offset
            + 1
            + voice_lpc_order
            + 1
            + noise_lpc_order :
        ] = 0


class SawSing(VocoderParameterEncoderInterface):
    def __init__(
        self,
        voice_n_mag: int,
        noise_n_mag: int,
        *args,
        extra_split_sizes: List[int] = [],
        kwargs: dict = {},
    ):
        super().__init__(
            *args,
            extra_split_sizes=extra_split_sizes + [voice_n_mag, noise_n_mag],
            **kwargs,
        )

    def forward(
        self, h: Tensor
    ) -> Tuple[
        Tuple[Tensor, Optional[Tensor]],
        Tuple[Any, ...],
        Tuple[Any, ...],
        Tuple[Any, ...],
        Tuple[Any, ...],
    ]:
        *f0_params, voice_log_mag, noise_log_mag = super().forward(h)

        return (
            f0_params,
            (),
            (voice_log_mag,),
            (noise_log_mag,),
            (),
        )


class MLSAEnc(VocoderParameterEncoderInterface):
    def __init__(
        self,
        sp_mcep_order: int,
        ap_mcep_order: int,
        *args,
        extra_split_sizes: List[int] = [],
        kwargs: dict = {},
    ):
        super().__init__(
            *args,
            extra_split_sizes=extra_split_sizes
            + [sp_mcep_order + 1, ap_mcep_order + 1],
            **kwargs,
        )

        self.backbone.out_linear.weight.data.zero_()
        self.backbone.out_linear.bias.data.zero_()

    def forward(
        self, h: Tensor
    ) -> Tuple[
        Tuple[Tensor, Optional[Tensor]],
        Tuple[Any, ...],
        Tuple[Any, ...],
        Tuple[Any, ...],
    ]:
        *f0_params, sp_mc, ap_mc_logits = super().forward(h)
        ap_mc = ap_mc_logits.sigmoid()

        return (
            f0_params,
            (),
            (ap_mc,),
            (sp_mc,),
        )
