import pytest
import torch

from models.synth import GlottalFlowTable


@pytest.mark.parametrize("table_type", ["flow", "derivative"])
@pytest.mark.parametrize("normalize_method", [None, "constant_power", "peak"])
@pytest.mark.parametrize("align_peak", [True, False])
def test_glottal_construction(table_type, normalize_method, align_peak):
    glottal = GlottalFlowTable(
        table_type=table_type, normalize_method=normalize_method, align_peak=align_peak
    )
    assert glottal is not None


def test_glottal_forward():
    hop_size = 80
    seq_length = 16010
    x = torch.linspace(0.1, 0.3, seq_length).cumsum(0) % 1
    x = x.unsqueeze(0)
    weight = torch.rand(1, seq_length // hop_size)
    glottal = GlottalFlowTable()

    y = glottal(x, weight, hop_size)
    assert y.shape == (1, seq_length)

    weight = torch.randn(1, seq_length // hop_size, 100).softmax(-1)
    y = glottal(x, weight, hop_size)
    assert y.shape == (1, seq_length)