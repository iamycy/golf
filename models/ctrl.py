from typing import Callable, List, Optional, Tuple, Union
import torch
from functools import reduce
from itertools import starmap

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


def wrap_ctrl_fn(
    split_size: Tuple[int, ...] = (), trsfm_fn: TRSFM_TYPE = lambda *x: ()
):
    def ctrl_fn(other_ctrl_fn: SPLIT_TRSFM_SIGNATURE):
        def split_and_trsfm(
            split_sizes: Tuple[Tuple[int, ...], ...],
            trsfm_fns: Tuple[TRSFM_TYPE, ...],
        ):
            return other_ctrl_fn(split_sizes + (split_size,), trsfm_fns + (trsfm_fn,))

        return split_and_trsfm

    return ctrl_fn


class Controllable(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.ctrl = wrap_ctrl_fn()


class PassThrough(Controllable):
    def forward(self, x: AudioTensor, *args, **kwargs) -> AudioTensor:
        return x


class Synth(torch.nn.Module):
    def get_split_sizes_and_trsfms(self):
        filtered_modules = list(
            filter(lambda x: isinstance(x[1], Controllable), self.named_children())
        )
        return reduce(
            lambda x, f: f(x),
            starmap(lambda _, x: x.ctrl, reversed(filtered_modules)),
            DUMMY_SPLIT_TRSFM,
        )((), ()) + (tuple(starmap(lambda x, _: x + "_params", filtered_modules)),)
