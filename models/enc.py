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
        extra_split_size: List[int],
        backbone_type: str,
        f0_min: float = 80,
        f0_max: float = 1000,
        **kwargs,
    ):
        super().__init__()
        self.split_size = [1] + extra_split_size
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
    ) -> Tuple[
        Tensor, Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...]
    ]:
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
        return (f0,) + tuple(_)


class GlottalComplexConjLPCEncoder(VocoderParameterEncoderInterface):
    def __init__(
        self,
        voice_lpc_order: int,
        noise_lpc_order: int,
        table_weight_hidden_size: int,
        *args,
        max_abs_value: float = 0.99,
        kwargs: dict = {},
    ):
        assert voice_lpc_order % 2 == 0
        assert noise_lpc_order % 2 == 0
        super().__init__(
            extra_split_size=[
                voice_lpc_order,
                1,
                noise_lpc_order,
                1,
                table_weight_hidden_size,
            ],
            *args,
            **kwargs,
        )
        self.logits2biquads = get_logits2biquads("conj", max_abs_value)
        self.backbone.out_linear.weight.data.zero_()
        self.backbone.out_linear.bias.data[
            1 : 1 + voice_lpc_order : 2
        ] = -10  # initialize magnitude close to zero
        self.backbone.out_linear.bias.data[1 + voice_lpc_order] = 0  # initialize gain
        self.backbone.out_linear.bias.data[
            1 + voice_lpc_order + 1 : 1 + voice_lpc_order + 1 + noise_lpc_order : 2
        ] = -10  # initialize magnitude close to zero
        self.backbone.out_linear.bias.data[2 + voice_lpc_order + noise_lpc_order] = -10

    def forward(
        self, h: Tensor
    ) -> Tuple[
        Tensor, Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...]
    ]:
        batch, frames, _ = h.shape
        (
            f0,
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

        voice_gain = voice_log_gain.squeeze(-1).exp()
        noise_gain = noise_log_gain.squeeze(-1).exp()

        return (
            f0,
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
        super().__init__(*args, **kwargs)
        self.logits2biquads = get_logits2biquads("coef", max_abs_value)
        self.backbone.out_linear.bias.data[1 : 1 + voice_lpc_order :] = 0
        self.backbone.out_linear.bias.data[
            1 + voice_lpc_order + 1 : 1 + voice_lpc_order + 1 + noise_lpc_order :
        ] = 0
