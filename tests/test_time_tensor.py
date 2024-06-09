import pytest
import torch

from models.utils import TimeTensor


def test_basic():
    x = TimeTensor(torch.randn(1, 100), hop_length=10, names=("B", "T"))
    assert x.size() == (1, 100)
    assert x.hop_length == 10

    x = x.align_to("T", "B")
    assert type(x) == TimeTensor, x
    assert x.shape == (100, 1)
    assert x.hop_length == 10
    assert x.names == ("T", "B")

    x1 = x.reduce_hop_length()
    assert type(x1) == TimeTensor, x1
    assert x1.shape == (991, 1)
    assert x1.hop_length == 1
    assert x1.names == ("T", "B")

    x2 = x.reduce_hop_length(5)
    assert type(x2) == TimeTensor, x2
    assert x2.shape == (496, 1)
    assert x2.hop_length == 2
    assert x2.names == ("T", "B")

    x3 = x1 + x2 * x
    assert x3.shape == (991, 1)
