import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from typing import Optional, Union, List, Tuple, Callable

from .utils import (
    get_transformed_lf,
    AudioTensor,
)


__all__ = [
    "OscillatorInterface",
    "IndexedGlottalFlowTable",
    "WeightedGlottalFlowTable",
    "DownsampledIndexedGlottalFlowTable",
    "DownsampledWeightedGlottalFlowTable",
]


def check_input_hook(m, args):
    upsampled_phase, *args = args
    assert upsampled_phase.ndim == 2, upsampled_phase.shape
    assert torch.all(upsampled_phase >= 0) and torch.all(upsampled_phase <= 0.5)


def check_output_hook(m, args, out):
    assert out.ndim == 2, out.shape
    assert isinstance(out, AudioTensor)
    assert out.hop_length == 1


def check_weight_hook(m, args):
    _, weight, *args = args
    assert torch.all(weight >= 0) and torch.all(weight <= 1)


class OscillatorInterface(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._input_handle = self.register_forward_pre_hook(check_input_hook)
        self._output_handle = self.register_forward_hook(check_output_hook)

    def forward(
        self,
        phase: AudioTensor,
        *args,
        **kwargs,
    ) -> AudioTensor:
        raise NotImplementedError


class GlottalFlowTable(OscillatorInterface):
    def __init__(
        self,
        table_size: int = 100,
        table_type: str = "derivative",
        normalize_method: str = "constant_power",
        align_peak: bool = True,
        trainable: bool = False,
        min_R_d: float = 0.3,
        max_R_d: float = 2.7,
        **kwargs,
    ):
        super().__init__()

        self.register_buffer(
            "R_d_values",
            torch.exp(torch.linspace(math.log(min_R_d), math.log(max_R_d), table_size)),
        )

        table = []
        for R_d in self.R_d_values:
            table.append(get_transformed_lf(R_d=R_d, **kwargs))

        table = torch.stack(table)
        if table_type == "flow":
            table = table.cumsum(dim=1)
        elif table_type == "derivative":
            pass
        else:
            raise ValueError(f"unknown table_type: {table_type}")

        if align_peak:
            # get peak position
            if table_type == "derivative":
                peak_pos = table.argmin(dim=1)
            else:
                peak_pos = table.argmax(dim=1)

            align_peak_pos = peak_pos.max().item()
            peak_pos = peak_pos.tolist()
            for i in range(table.shape[0]):
                table[i] = torch.roll(table[i], align_peak_pos - peak_pos[i])

        if normalize_method == "constant_power":
            # normalize to constant power
            table = table / table.norm(dim=1, keepdim=True) * math.sqrt(table.shape[1])
        elif normalize_method == "peak":
            if table_type == "flow":
                # normalize to peak 1
                table = table / table.max(dim=1, keepdim=True).values
        elif normalize_method is None:
            pass
        else:
            raise ValueError(f"unknown normalize_method: {normalize_method}")

        if trainable:
            self.register_parameter("table", nn.Parameter(table))
        else:
            self.register_buffer("table", table)

        # self._table_weight_handle = self.register_forward_pre_hook(check_weight_hook)

    @staticmethod
    def generate(wrapped_phase: AudioTensor, tables: AudioTensor) -> AudioTensor:
        """
        Args:
            wrapped_phase: (batch, seq_len)
            tables: (batch, seq_len / hop_length + 1, table_size)
            hop_length: int
        """
        assert wrapped_phase.hop_length == 1
        wrapped_phase = wrapped_phase.as_tensor()
        batch, seq_len = wrapped_phase.shape
        hop_length = tables.hop_length
        tables = tables.as_tensor()

        # pad phase to have multiple of hop_length
        pad_length = (hop_length - seq_len % hop_length) % hop_length
        wrapped_phase = F.pad(wrapped_phase, (0, pad_length), "replicate")
        wrapped_phase = wrapped_phase.view(batch, -1, hop_length)

        # make sure flow has seq_len / hop_length + 1 frames
        if tables.shape[1] < wrapped_phase.shape[1] + 1:
            tables = F.pad(
                tables,
                (0, 0, 0, wrapped_phase.shape[1] - tables.shape[1] + 1),
                "replicate",
            )
        else:
            tables = tables[:, : wrapped_phase.shape[1] + 1]

        table_length = tables.shape[2]

        table_index_raw = wrapped_phase * table_length
        floor_index = table_index_raw.long().clip_(0, table_length - 1)

        # shape = (batch, seq_len / hop_length, hop_length)
        p = table_index_raw - floor_index

        # shape = (batch, seq_len / hop_length + 1, table_length + 1)
        padded_tables = torch.cat([tables, tables[:, :, :1]], dim=2)
        floor_flow = padded_tables[:, :-1]
        ceil_flow = padded_tables[:, 1:]
        p2 = (
            torch.arange(
                hop_length, device=wrapped_phase.device, dtype=wrapped_phase.dtype
            )
            / hop_length
        )

        # first, pick floor flow
        # create dummy index to select floor flow
        dummy_index_0 = (
            torch.arange(batch, device=wrapped_phase.device)
            .view(-1, 1, 1)
            .repeat(1, wrapped_phase.shape[1], hop_length)
            .flatten()
        )
        dummy_index_1 = (
            torch.arange(wrapped_phase.shape[1], device=wrapped_phase.device)
            .view(1, -1, 1)
            .repeat(batch, 1, hop_length)
            .flatten()
        )
        dummy_index_2 = floor_index.flatten()
        selected_floor_flow = (
            floor_flow[dummy_index_0, dummy_index_1, dummy_index_2].view(
                *wrapped_phase.shape
            )
            * (1 - p)
            + floor_flow[dummy_index_0, dummy_index_1, dummy_index_2 + 1].view(
                *wrapped_phase.shape
            )
            * p
        )

        # second, pick ceil flow
        selected_ceil_flow = (
            ceil_flow[dummy_index_0, dummy_index_1, dummy_index_2].view(
                *wrapped_phase.shape
            )
            * (1 - p)
            + ceil_flow[dummy_index_0, dummy_index_1, dummy_index_2 + 1].view(
                *wrapped_phase.shape
            )
            * p
        )
        final_flow = selected_floor_flow * (1 - p2) + selected_ceil_flow * p2
        final_flow = final_flow.view(batch, -1)[:, :seq_len]
        return AudioTensor(final_flow)

    def forward(
        self,
        phase: AudioTensor,
        table_select_weight: AudioTensor,
        *args,
    ) -> AudioTensor:
        """
        input:
            upsampled_phase: (batch, seq_len)
            table_select_weight: (batch, seq_len / hop_length, ...)
        """

        raise NotImplementedError


class IndexedGlottalFlowTable(GlottalFlowTable):
    def forward(
        self,
        phase: AudioTensor,
        table_select_weight: AudioTensor,
        phase_offset: AudioTensor = None,
    ) -> Tensor:
        assert table_select_weight.dim() == 2
        assert torch.all(table_select_weight >= 0) and torch.all(
            table_select_weight <= 1
        )
        num_tables, table_length = self.table.shape
        table_index_raw = table_select_weight * (num_tables - 1)
        floor_index = table_index_raw.as_tensor().long().clip_(0, num_tables - 2)
        p = table_index_raw - floor_index
        p = torch.unsqueeze(p, -1)
        interp_tables = (
            self.table[floor_index.flatten()].view(
                floor_index.shape[0], floor_index.shape[1], table_length
            )
            * (1 - p)
            + self.table[floor_index.flatten() + 1].view(
                floor_index.shape[0], floor_index.shape[1], table_length
            )
            * p
        )
        # interp_tables = table_select_weight.new_tensor(interp_tables)
        upsampled_phase = phase.reduce_hop_length()
        instant_phase = torch.cumsum(upsampled_phase, 1)
        if phase_offset is not None:
            instant_phase = instant_phase + phase_offset
        wrapped_phase = instant_phase % 1
        return self.generate(wrapped_phase, interp_tables)


class WeightedGlottalFlowTable(GlottalFlowTable):
    def forward(
        self,
        phase: AudioTensor,
        table_select_weight: AudioTensor,
        phase_offset: AudioTensor = None,
    ) -> Tensor:
        assert table_select_weight.dim() == 3
        assert table_select_weight.shape[2] == self.table.shape[0]
        assert torch.all(table_select_weight >= 0) and torch.all(
            table_select_weight <= 1
        )
        weighted_tables = table_select_weight @ self.table
        instant_phase = torch.cumsum(phase.reduce_hop_length(), 1)
        if phase_offset is not None:
            instant_phase = instant_phase + phase_offset
        wrapped_phase = instant_phase % 1
        return self.generate(wrapped_phase, weighted_tables)


def get_downsampler(hop_rate: int, in_channels: int, output_channels: int):
    return nn.Sequential(
        nn.AvgPool1d(
            kernel_size=hop_rate,
            stride=hop_rate,
            padding=hop_rate // 2,
        ),
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=1,
        ),
        nn.GLU(dim=1),
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=1,
        ),
    )


class DownsampledIndexedGlottalFlowTable(IndexedGlottalFlowTable):
    hop_rate: int

    def __init__(
        self,
        hop_rate: int,
        in_channels: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hop_rate = hop_rate
        self.model = get_downsampler(hop_rate, in_channels, 1)

    def forward(
        self,
        phase: AudioTensor,
        h: AudioTensor,
        *args,
    ) -> AudioTensor:
        """
        input:
            upsampled_phase: (batch, seq_len)
            h: (batch, frames, in_channels)
        """
        table_control = (
            self.model(torch.transpose(h.as_tensor(), 1, 2)).squeeze(1).sigmoid()
        )
        table_control = AudioTensor(
            table_control, hop_length=h.hop_length * self.hop_rate
        )
        return super().forward(phase, table_control, *args)


class DownsampledWeightedGlottalFlowTable(WeightedGlottalFlowTable):
    hop_rate: int

    def __init__(
        self,
        hop_rate: int,
        in_channels: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hop_rate = hop_rate
        self.model = get_downsampler(hop_rate, in_channels, self.table.shape[0])

    def forward(
        self,
        phase: AudioTensor,
        h: AudioTensor,
        *args,
    ) -> AudioTensor:
        """
        input:
            upsampled_phase: (batch, seq_len)
            h: (batch, frames, in_channels)
        """
        table_control = self.model(h.transpose(1, 2)).softmax(dim=1).transpose(1, 2)
        table_control = AudioTensor(
            table_control,
            hop_length=h.hop_length * self.hop_rate,
        )
        return super().forward(phase, table_control, *args)


class HarmonicOscillator(OscillatorInterface):
    """synthesize audio with a bank of harmonic oscillators"""

    def forward(
        self,
        phase: AudioTensor,
        amplitudes: AudioTensor,
        initial_phase: Optional[Union[Tensor, AudioTensor]] = None,
        phase_offset: Optional[AudioTensor] = None,
    ) -> AudioTensor:
        """
                   f0: B x T (Hz)
           amplitudes: B x T / hop_length x n_harmonic
        initial_phase: B x n_harmonic
         ---
             signal: B x T
        final_phase: B x 1 x 1
        """

        # harmonic synth
        assert phase.hop_length == 1
        n_harmonic = amplitudes.shape[-1]
        harm_series = torch.arange(1, n_harmonic + 1).to(phase.device)
        harmonics = phase.unsqueeze(-1) * harm_series
        harmonics = harmonics.reduce_hop_length()
        alias_mask = harmonics >= 0.5

        phase = torch.cumsum(harmonics, axis=1)
        if phase_offset is not None:
            # upsampled_phase_offset = phase_offset.align_to("B", "T").reduce_hop_length()
            # upsampled_phase_offset = upsampled_phase_offset % 1
            phase = phase + phase_offset.unsqueeze(-1) * harm_series

        if initial_phase is not None:
            phase = phase + initial_phase.unsqueeze(1)

        amplitudes = amplitudes.reduce_hop_length()

        # anti-aliasing
        amplitudes = torch.where(alias_mask, 0, amplitudes)

        # signal
        return (
            (torch.sin(phase * 2 * torch.pi)[..., None, :] @ amplitudes[..., None])
            .squeeze(-1)
            .squeeze(-1)
        )


class SawToothOscillator(HarmonicOscillator):
    """synthesize audio with a bank of sawtooth oscillators"""

    def __init__(self, num_harmonics: int, gain: float = 0.4) -> None:
        super().__init__()
        self.gain = gain
        self.register_buffer("amplitudes", 1 / torch.arange(1, num_harmonics + 1))

    def forward(
        self,
        phase: AudioTensor,
        initial_phase: Optional[Union[Tensor, AudioTensor]] = None,
        phase_offset: Optional[AudioTensor] = None,
        **kwargs,
    ) -> AudioTensor:

        amplitudes = self.amplitudes[None, None, :].repeat(*phase.shape, 1)
        return super().forward(phase, amplitudes, initial_phase, phase_offset)


class PulseTrain(OscillatorInterface):
    def forward(self, phase: AudioTensor, phase_offset: AudioTensor) -> AudioTensor:
        # Make mask represents voiced region.
        upsampled_phase = phase.reduce_hop_length()
        instant_phase = torch.cumsum(upsampled_phase, dim=1)
        if phase_offset is not None:
            instant_phase = instant_phase + phase_offset
        wrapped_phase = instant_phase % 1
        phase_transition = (wrapped_phase[:, 1:] - wrapped_phase[:, :-1]) < 0
        out = torch.zeros_like(upsampled_phase)
        out[:, 1:][phase_transition] = upsampled_phase[:, 1:][phase_transition].rsqrt()
        return AudioTensor(out)


class AdditivePulseTrain(HarmonicOscillator):
    def __init__(self, num_harmonics: int = 155) -> None:
        super().__init__()
        self.num_harmonics = num_harmonics

    def forward(
        self,
        phase: AudioTensor,
        initial_phase: Optional[Union[Tensor, AudioTensor]] = None,
        phase_offset: Optional[AudioTensor] = None,
        **kwargs,
    ) -> AudioTensor:
        amplitudes = (
            torch.ones_like(phase).unsqueeze(-1).repeat(1, 1, self.num_harmonics)
        )
        num_freq_bins = 0.5 / phase
        amplitudes = amplitudes * num_freq_bins.unsqueeze(-1).rsqrt()
        return super().forward(phase, amplitudes, initial_phase, phase_offset)
