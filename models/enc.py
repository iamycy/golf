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
        self.out_linear.weight.data.zero_()
        self.out_linear.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.out_linear(x)


class VocoderParameterEncoderInterface(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        learn_voicing: bool = False,
        f0_min: float = 80,
        f0_max: float = 1000,
        split_sizes: Tuple[Tuple[int, ...], ...] = (),
        trsfms: Tuple[Callable[..., Tuple[torch.Tensor, ...]], ...] = (),
        **kwargs,
    ):
        super().__init__()
        self.split_sizes = split_sizes
        self.trsfms = trsfms
        concated_split_sizes = sum(split_sizes, ())
        self._split_size = ((1, 1) if learn_voicing else (1,)) + concated_split_sizes
        self.learn_voicing = learn_voicing
        self.log_f0_min = math.log(f0_min)
        self.log_f0_max = math.log(f0_max)

        module_path, class_name = backbone_type.rsplit(".", 1)
        module = import_module(module_path)

        self.backbone = getattr(module, class_name)(
            out_channels=sum(self._split_size), **kwargs
        )
        # self.backbone = torch.compile(self.backbone)

    def logits2f0(self, logits: AudioTensor) -> AudioTensor:
        return torch.exp(
            torch.sigmoid(logits) * (self.log_f0_max - self.log_f0_min)
            + self.log_f0_min
        )

    def forward(
        self, h: AudioTensor
    ) -> Tuple[AudioTensor, ...,]:
        f0_logits, *_ = [
            h.new_tensor(torch.squeeze(t, 2))
            for t in self.backbone(h.as_tensor()).split(self._split_size, dim=2)
        ]
        f0 = self.logits2f0(f0_logits)
        if self.learn_voicing:
            voicing, *_ = _
        else:
            voicing = None

        groupped_logits = []
        for splits in self.split_sizes:
            groupped_logits.append(_[: len(splits)])
            _ = _[len(splits) :]

        transformed = map(lambda x: x[0](*x[1]), zip(self.trsfms, groupped_logits))
        return (f0,) + tuple(transformed) + (voicing,)
