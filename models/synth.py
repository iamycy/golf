import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from typing import Optional, Union, List, Tuple, Callable

from .utils import get_transformed_lf, get_radiation_time_filter, get_window_fn
from .lpc import BatchLPCSynth, BatchSecondOrderLPCSynth


class GlottalFlowTable(nn.Module):
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

    def forward(
        self, wrapped_phase: Tensor, table_select_weight: Tensor, hop_size: int
    ) -> Tensor:
        """
        input:
            wrapped_phase: (batch, seq_len)
            table_select_weight: (batch, seq_len / hop_size + 1) or (batch, seq_len / hop_size + 1, table_size)
        """

        assert torch.all(table_select_weight >= 0) and torch.all(
            table_select_weight <= 1
        )
        assert torch.all(wrapped_phase >= 0) and torch.all(wrapped_phase <= 1)

        batch, seq_len = wrapped_phase.shape
        table_length = self.table.shape[1]

        if table_select_weight.dim() == 2:
            table_index_raw = table_select_weight * (self.table.shape[0] - 1)
            floor_index = table_index_raw.long().clip_(0, self.table.shape[0] - 2)
            p = table_index_raw - floor_index
            p = p.unsqueeze(-1)
            flow = (
                self.table[floor_index.flatten()].view(
                    floor_index.shape[0], floor_index.shape[1], table_length
                )
                * (1 - p)
                + self.table[floor_index.flatten() + 1].view(
                    floor_index.shape[0], floor_index.shape[1], table_length
                )
                * p
            )
        elif table_select_weight.dim() == 3:
            assert table_select_weight.shape[2] == self.table.shape[0]
            flow = table_select_weight @ self.table
        else:
            raise ValueError("table_select_weight must be 2 or 3 dim")

        # pad phase to have multiple of hop_size
        pad_len = (hop_size - seq_len % hop_size) % hop_size
        wrapped_phase = F.pad(
            wrapped_phase.unsqueeze(1), (0, pad_len), "replicate"
        ).squeeze(1)
        wrapped_phase = wrapped_phase.view(batch, -1, hop_size)

        # make sure flow has seq_len / hop_size + 1 frames
        if flow.shape[1] < wrapped_phase.shape[1] + 1:
            flow = F.pad(
                flow, (0, 0, 0, wrapped_phase.shape[1] - flow.shape[1] + 1), "replicate"
            )
        else:
            flow = flow[:, : wrapped_phase.shape[1] + 1]

        table_index_raw = wrapped_phase * table_length
        floor_index = table_index_raw.long().clip_(0, table_length - 1)

        # shape = (batch, seq_len / hop_size, hop_size)
        p = table_index_raw - floor_index

        # shape = (batch, seq_len / hop_size + 1, table_length + 1)
        padded_flow = torch.cat([flow, flow[:, :, :1]], dim=2)
        floor_flow = padded_flow[:, :-1]
        ceil_flow = padded_flow[:, 1:]
        p2 = (
            torch.arange(
                hop_size, device=wrapped_phase.device, dtype=wrapped_phase.dtype
            )
            / hop_size
        )

        # first, pick floor flow
        # create dummy index to select floor flow
        dummy_index_0 = (
            torch.arange(batch, device=wrapped_phase.device)
            .view(-1, 1, 1)
            .repeat(1, wrapped_phase.shape[1], hop_size)
            .flatten()
        )
        dummy_index_1 = (
            torch.arange(wrapped_phase.shape[1], device=wrapped_phase.device)
            .view(1, -1, 1)
            .repeat(batch, 1, hop_size)
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

        return final_flow


class GlottalSynth(nn.Module):
    def __init__(
        self,
        hop_length: int,
        wavetable_hop_length: int,
        window_size: int = None,
        window: str = "hann",
        apply_radiation: bool = False,
        radiation_kernel_size: int = 256,
        **kwargs,
    ):

        super().__init__()
        window_fn = get_window_fn(window)
        self.hop_length = hop_length
        self.wavetable_hop_length = wavetable_hop_length
        self.lpc = BatchLPCSynth(hop_length, window_size, window_fn)
        # self.lpc = BatchSecondOrderLPCSynth(hop_length, window_size, window_fn)
        self.glottal = GlottalFlowTable(**kwargs)

        if apply_radiation:
            self.register_buffer(
                "radiation_filter",
                get_radiation_time_filter(radiation_kernel_size, window_fn=window_fn)
                .flip(0)
                .unsqueeze(0)
                .unsqueeze(0),
            )
            self.radiation_filter_padding = self.radiation_filter.shape[-1] // 2

    def linear_upsample(self, x: Tensor):
        length = x.size(1)
        out_length = (length - 1) * self.hop_length + 1
        return F.interpolate(
            x.unsqueeze(1), out_length, mode="linear", align_corners=True
        ).squeeze(1)

    def forward(
        self,
        instant_freq: Tensor,
        phase_offset: Tensor,
        # log_snr: Tensor,
        table_select_weight: Tensor,
        log_gains: Tensor,
        vocal_tract_coeffs: Tensor,
        log_noise_gain: Tensor,
        noise_filter_coeffs: Tensor,
    ) -> Tensor:
        assert torch.all(instant_freq >= 0) and torch.all(instant_freq <= 0.5)
        assert torch.all(phase_offset >= 0) and torch.all(phase_offset <= 1)

        # instant_freq = self.linear_upsample(instant_freq)
        phase = torch.cumsum(instant_freq, dim=1)

        phase_offset_diff = phase_offset.diff(dim=1)
        # wrapp the differences into [-0.5, 0.5]
        phase_offset_diff = (phase_offset_diff + 0.5) % 1 - 0.5
        phase_offset_diff = torch.cat([phase_offset[:, :1], phase_offset_diff], dim=1)
        phase_offset = torch.cumsum(phase_offset_diff, dim=1)
        phase_offset = self.linear_upsample(phase_offset)

        phase = phase[:, : phase_offset.shape[1]] + phase_offset[:, : phase.shape[1]]
        phase = phase % 1

        glottal_flow = self.glottal(
            phase, table_select_weight, self.wavetable_hop_length
        )

        # mix with white noise
        # log_snr = self.linear_upsample(log_snr)[:, : glottal_flow.shape[1]]
        # noise = torch.randn_like(glottal_flow)
        # alpha = log_snr.sigmoid().sqrt()
        # sigma = torch.sigmoid(-log_snr).sqrt()
        # # alpha = (instant_freq > 0.00125).float()
        # # sigma = 1 - alpha
        # excitation = alpha * glottal_flow + sigma * noise

        # vocal tract filter
        gain = torch.exp(log_gains)
        lpc_coeffs = torch.cat([gain.unsqueeze(-1), vocal_tract_coeffs], dim=-1)
        y = self.lpc(glottal_flow, lpc_coeffs)
        # y = self.lpc(glottal_flow, vocal_tract_coeffs, gain)

        # noise filter
        noise_gain = torch.exp(log_noise_gain)
        noise_lpc_coeffs = torch.cat(
            [noise_gain.unsqueeze(-1), noise_filter_coeffs], dim=-1
        )
        noise = self.lpc(torch.randn_like(glottal_flow), noise_lpc_coeffs)
        # noise = self.lpc(torch.randn_like(glottal_flow), noise_filter_coeffs, noise_gain)

        y = y + noise

        if hasattr(self, "radiation_filter"):
            y = F.conv1d(
                y.unsqueeze(1),
                self.radiation_filter,
                padding=self.radiation_filter_padding,
            ).squeeze(1)

        return y
