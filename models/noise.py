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


class SignFLipNoise(NoiseInterface):
    def __init__(self):
        super().__init__(torch.distributions.Uniform(-1, 1))

    def forward(self, ref: Tensor, *args, **kwargs) -> Tensor:
        sign = ref.new_empty(ref.shape[:-1]).uniform_(-1, 1).sign()
        tmp = torch.ones_like(ref)
        tmp[..., ::2] = sign.unsqueeze(-1)
        tmp[..., 1::2] = -sign.unsqueeze(-1)
        return tmp
