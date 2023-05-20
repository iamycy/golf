import torch
from torch import nn, Tensor
import math
from importlib import import_module
from typing import Optional, Union, List, Tuple, Callable, Any

from .utils import get_logits2biquads, biquads2lpc, AudioTensor, rc2lpc


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
        # self.backbone = torch.compile(self.backbone)

    def logits2f0(self, logits: Tensor) -> Tensor:
        return torch.exp(
            logits.sigmoid() * (self.log_f0_max - self.log_f0_min) + self.log_f0_min
        )

    def forward(
        self, h: AudioTensor
    ) -> Tuple[AudioTensor, ...,]:
        """
        Args:
            h: (batch_size, frames, features)
        Returns:
            harm_osc_params: Tuple[Tensor, ...]
            harm_filt_params: Tuple[Tensor, ...]
            noise_filt_params: Tuple[Tensor, ...]
            noise_params: Tuple[Tensor, ...]
        """
        f0_logits, *_ = self.backbone(h.as_tensor()).split(self.split_size, dim=-1)
        f0 = self.logits2f0(f0_logits).squeeze(-1)
        f0 = h.new_tensor(f0)
        if self.learn_voicing:
            voicing = f0.new_tensor(_[0].squeeze(-1))
            _ = _[1:]
        else:
            voicing = None
        return (f0,) + tuple(h.new_tensor(x) for x in _) + (voicing,)


class GlottalComplexConjLPCEncoder(VocoderParameterEncoderInterface):
    def __init__(
        self,
        table_weight_hidden_size: int,
        lpc_order: int,
        noise_n_mag: int,
        *args,
        max_abs_value: float = 0.99,
        extra_split_sizes: List[int] = [],
        kwargs: dict = {},
    ):
        assert lpc_order % 2 == 0

        extra_split_sizes.extend(
            [
                table_weight_hidden_size,
                lpc_order,
                1,
                noise_n_mag,
            ]
        )
        super().__init__(
            *args,
            extra_split_sizes=extra_split_sizes,
            **kwargs,
        )
        self.logits2biquads = get_logits2biquads("conj", max_abs_value)
        self.backbone.out_linear.weight.data.zero_()
        self.backbone.out_linear.bias.data.zero_()

        offset = 1 if self.learn_voicing else 0
        offset += table_weight_hidden_size
        self.backbone.out_linear.bias.data[
            offset + 1 : offset + 1 + lpc_order : 2
        ] = -10  # initialize magnitude close to zero
        self.backbone.out_linear.bias.data[
            offset + 1 + lpc_order
        ] = 0  # initialize gain

    def forward(
        self, h: AudioTensor
    ) -> Tuple[
        AudioTensor,
        Tuple[Any, ...],
        Tuple[Any, ...],
        Tuple[Any, ...],
        Tuple[Any, ...],
        Union[None, AudioTensor],
    ]:
        batch, frames, _ = h.shape
        (
            f0,
            h,
            lpc_logits,
            log_gain,
            log_mag,
            voicing,
        ) = super().forward(h)
        voice_biquads = self.logits2biquads(
            lpc_logits.as_tensor().view(batch, frames, -1, 2)
        )
        lpc_coeffs = biquads2lpc(voice_biquads)
        lpc_coeffs = lpc_logits.new_tensor(lpc_coeffs)

        gain = torch.exp(torch.squeeze(log_gain, 2))

        return (
            f0,
            (h,),
            (),
            (log_mag,),
            (gain, lpc_coeffs),
            voicing,
        )


class GlottalRealCoeffLPCEncoder(GlottalComplexConjLPCEncoder):
    def __init__(
        self,
        table_weight_hidden_size: int,
        lpc_order: int,
        noise_n_mag: int,
        *args,
        max_abs_value: float = 0.99,
        **kwargs,
    ):
        super().__init__(
            table_weight_hidden_size,
            lpc_order,
            noise_n_mag,
            *args,
            max_abs_value=max_abs_value,
            **kwargs,
        )
        self.logits2biquads = get_logits2biquads("coef", max_abs_value)
        self.backbone.out_linear.bias.data.zero_()


class GlottalRC2LPCEncoder(VocoderParameterEncoderInterface):
    def __init__(
        self,
        table_weight_hidden_size: int,
        lpc_order: int,
        noise_n_mag: int,
        *args,
        max_abs_value: float = 0.99,
        extra_split_sizes: List[int] = [],
        kwargs: dict = {},
    ):

        extra_split_sizes.extend(
            [
                table_weight_hidden_size,
                lpc_order,
                1,
                noise_n_mag,
            ]
        )
        super().__init__(
            *args,
            extra_split_sizes=extra_split_sizes,
            **kwargs,
        )
        self.backbone.out_linear.weight.data.zero_()
        self.backbone.out_linear.bias.data.zero_()
        self.max_abs_value = max_abs_value

    def forward(
        self, h: AudioTensor
    ) -> Tuple[
        AudioTensor,
        Tuple[Any, ...],
        Tuple[Any, ...],
        Tuple[Any, ...],
        Tuple[Any, ...],
        Union[None, AudioTensor],
    ]:
        (
            f0,
            h,
            lpc_logits,
            log_gain,
            log_mag,
            voicing,
        ) = super().forward(h)

        lpc_coeffs = h.new_tensor(
            rc2lpc(lpc_logits.as_tensor().tanh() * self.max_abs_value)
        )
        gain = torch.exp(torch.squeeze(log_gain, 2))

        return (
            f0,
            (h,),
            (),
            (log_mag,),
            (gain, lpc_coeffs),
            voicing,
        )


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


class DDSPAdd(VocoderParameterEncoderInterface):
    def __init__(
        self,
        num_harmonics: int,
        noise_n_mag: int,
        *args,
        extra_split_sizes: List[int] = [],
        kwargs: dict = {},
    ):
        super().__init__(
            *args,
            extra_split_sizes=extra_split_sizes + [1, num_harmonics, noise_n_mag],
            **kwargs,
        )

        self.backbone.out_linear.weight.data.zero_()
        offset = 1 if self.learn_voicing else 0
        self.backbone.out_linear.bias.data[
            offset : offset + 1 + num_harmonics
        ] = 0  # initialize f0
        self.backbone.out_linear.bias.data[
            offset + 1 + num_harmonics :
        ] = -10  # initialize magnitude close to zero

    def forward(
        self, h: Tensor
    ) -> Tuple[
        Tuple[Tensor, Optional[Tensor]],
        Tuple[Any, ...],
        Tuple[Any, ...],
        Tuple[Any, ...],
        Tuple[Any, ...],
    ]:
        *f0_params, log_loudness, harmonics_logits, noise_log_mag = super().forward(h)
        loudness = log_loudness.exp()
        # harmonics = harmonics_logits.softmax(-1)
        harmonics = harmonics_logits.sigmoid()
        harmonics = harmonics / harmonics.sum(-1, keepdim=True)
        amplitudes = harmonics * loudness

        return (
            f0_params,
            (amplitudes,),
            (),
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
        f0, sp_mc, ap_mc_logits, voicing = super().forward(h)
        ap_mc = torch.sigmoid(ap_mc_logits)

        return (
            f0,
            (),
            (),
            (ap_mc,),
            (sp_mc,),
            voicing,
        )


class PulseTrainRealCoeffLPCEncoder(VocoderParameterEncoderInterface):
    def __init__(
        self,
        voice_lpc_order: int,
        noise_lpc_order: int,
        *args,
        max_abs_value: float = 0.99,
        extra_split_sizes: List[int] = [],
        kwargs: dict = {},
    ):
        super().__init__(
            *args,
            extra_split_sizes=extra_split_sizes
            + [voice_lpc_order, 1, noise_lpc_order, 1],
            **kwargs,
        )

        self.logits2biquads = get_logits2biquads("coef", max_abs_value)
        self.backbone.out_linear.weight.data.zero_()
        self.backbone.out_linear.bias.data.zero_()
        self.backbone.out_linear.bias.data[-1] = -10  # initialize noise gain

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
        ) = super().forward(h)
        voice_biquads = self.logits2biquads(voice_lpc_logits.view(batch, frames, -1, 2))
        noise_biquads = self.logits2biquads(noise_lpc_logits.view(batch, frames, -1, 2))
        voice_lpc_coeffs = biquads2lpc(voice_biquads)
        noise_lpc_coeffs = biquads2lpc(noise_biquads)

        voice_gain = voice_log_gain.squeeze(-1).exp()
        noise_gain = noise_log_gain.squeeze(-1).exp()

        return (
            f0_params,
            (),
            (voice_gain, voice_lpc_coeffs),
            (noise_gain, noise_lpc_coeffs),
            (),
        )
