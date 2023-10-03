from typing import Callable, List, Optional, Tuple, Union
import torch
from functools import reduce

from .utils import AudioTensor

TRSFM_TYPE = Callable[..., Tuple[AudioTensor, ...]]

SPLIT_TRSFM_SIGNATURE = Callable[
    [Tuple[Tuple[int, ...], ...], Tuple[TRSFM_TYPE, ...]],
    Tuple[Tuple[Tuple[int, ...], ...], Tuple[TRSFM_TYPE, ...]],
]
DUMMY_SPLIT_TRSFM: SPLIT_TRSFM_SIGNATURE = lambda split_sizes, trsfm_fns: (
    split_sizes,
    trsfm_fns,
)


def default_ctrl_fn(other_ctrl_fn: SPLIT_TRSFM_SIGNATURE):
    def split_and_trsfm(
        split_sizes: Tuple[Tuple[int, ...], ...],
        trsfm_fns: Tuple[TRSFM_TYPE, ...],
    ):
        split_sizes = split_sizes + ((),)
        trsfm_fns = trsfm_fns + (lambda *x: (),)
        return other_ctrl_fn(split_sizes, trsfm_fns)

    return split_and_trsfm


class Controllable(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.ctrl = default_ctrl_fn


class PassThrough(Controllable):
    def forward(self, x: AudioTensor, *args, **kwargs) -> AudioTensor:
        return x


class Synth(torch.nn.Module):
    def get_split_sizes_and_trsfms(self):
        filtered_modules = [
            (name, m)
            for name, m in self.named_children()
            if isinstance(m, Controllable)
        ]
        param_keys = map(lambda x: x[0] + "_params", filtered_modules)

        split_trsfm = reduce(
            lambda x, f: f(x),
            map(lambda x: x[1].ctrl, filtered_modules[::-1]),
            DUMMY_SPLIT_TRSFM,
        )
        return split_trsfm((), ()) + (tuple(param_keys),)
