import torch
from torch import nn, Tensor
from typing import Optional, Union, List, Tuple, Callable


__all__ = [
    "NoiseInterface",
    "StandardNormalNoise",
    "UniformNoise",
]


class NoiseInterface(nn.Module):
    dist: torch.distributions.Distribution

    def __init__(self, dist: torch.distributions.Distribution):
        super().__init__()
        self.dist = dist

    def forward(self, ref: Tensor, *args, **kwargs) -> Tensor:
        return self.dist.sample(ref.shape).to(ref.device)


class StandardNormalNoise(NoiseInterface):
    def __init__(self):
        super().__init__(torch.distributions.Normal(0, 1))

    def forward(self, ref: Tensor, *args, **kwargs) -> Tensor:
        return torch.randn_like(ref)


class UniformNoise(NoiseInterface):
    def __init__(self):
        super().__init__(torch.distributions.Uniform(-1, 1))

    def forward(self, ref: Tensor, *args, **kwargs) -> Tensor:
        return torch.rand_like(ref) * 2 - 1
