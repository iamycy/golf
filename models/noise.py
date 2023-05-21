import torch
from torch import nn, Tensor
from typing import Optional, Union, List, Tuple, Callable
import math

from .ctrl import Controllable
from .utils import AudioTensor

__all__ = [
    "NoiseInterface",
    "StandardNormalNoise",
    "UniformNoise",
]


class NoiseInterface(Controllable):
    dist: torch.distributions.Distribution

    def __init__(self, dist: torch.distributions.Distribution):
        super().__init__()
        self.dist = dist

    def forward(self, ref: AudioTensor, *args, **kwargs) -> AudioTensor:
        return self.dist.sample(ref.shape).to(ref.device)


class StandardNormalNoise(NoiseInterface):
    def __init__(self):
        super().__init__(torch.distributions.Normal(0, 1))

    def forward(self, ref: AudioTensor, *args, **kwargs) -> AudioTensor:
        return torch.randn_like(ref)


class UniformNoise(NoiseInterface):
    def __init__(self):
        super().__init__(torch.distributions.Uniform(-math.sqrt(3), math.sqrt(3)))

    def forward(self, ref: AudioTensor, *args, **kwargs) -> AudioTensor:
        return (torch.rand_like(ref) - 0.5) * 2 * math.sqrt(3)


class SignFlipNoise(NoiseInterface):
    def __init__(self):
        super().__init__(torch.distributions.Uniform(-1, 1))

    def forward(self, ref: AudioTensor, *args, **kwargs) -> AudioTensor:
        sign = ref.as_tensor().new_empty(ref.shape[:-1]).uniform_(-1, 1).sign()
        tmp = torch.ones_like(ref).as_tensor()
        tmp[..., ::2] = sign.unsqueeze(-1)
        tmp[..., 1::2] = -sign.unsqueeze(-1)
        return ref.new_tensor(tmp)
