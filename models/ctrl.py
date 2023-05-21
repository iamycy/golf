from typing import Callable, List, Optional, Tuple, Union
import torch

TRSFM_TYPE = Callable[..., Tuple[torch.Tensor, ...]]

SPLIT_TRSFM_SIGNATURE = Callable[
    [Tuple[Tuple[int, ...], ...], Tuple[TRSFM_TYPE, ...]],
    Tuple[Tuple[Tuple[int, ...], ...], Tuple[TRSFM_TYPE, ...]],
]
DUMMY_SPLIT_TRSFM: SPLIT_TRSFM_SIGNATURE = lambda split_sizes, trsfm_fns: (
    split_sizes,
    trsfm_fns,
)


class Controllable(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def ctrl_fn(other_ctrl_fn: SPLIT_TRSFM_SIGNATURE):
            def split_and_trsfm(
                split_sizes: Tuple[Tuple[int, ...], ...],
                trsfm_fns: Tuple[TRSFM_TYPE, ...],
            ):
                split_sizes = split_sizes + ((),)
                trsfm_fns = trsfm_fns + (lambda *x: (),)
                return other_ctrl_fn(split_sizes, trsfm_fns)

            return split_and_trsfm

        self.ctrl = ctrl_fn
