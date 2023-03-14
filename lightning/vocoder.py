import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Tuple, Callable, Union
from torchaudio.transforms import MelSpectrogram
import random

from models.synth import GlottalSynth, GlottalFlowTable
from models.lpc import LPCSynth, BatchLPCSynth, BatchSecondOrderLPCSynth
from models.mel import Mel2Control
from models.utils import get_window_fn, get_radiation_time_filter


class WrappedMelSpectrogram(MelSpectrogram):
    def __init__(self, window: str, **kwargs):
        super().__init__(window_fn=get_window_fn(window), **kwargs)


class MelVocoderCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "model.hop_length", "model.mel_trsfm.init_args.hop_length"
        )
        parser.link_arguments("model.hop_length", "model.lpc.init_args.hop_length")
        # parser.link_arguments(
        #     (
        #         "model.hop_length",
        #         "model.table_hop_rate",
        #     ),
        #     "model.glottal_synth.init_args.wavetable_hop_length",
        #     compute_fn=lambda hop_length, table_hop_rate: hop_length * table_hop_rate,
        # )

        parser.link_arguments(
            "model.sample_rate", "model.mel_trsfm.init_args.sample_rate"
        )
        parser.link_arguments(
            "model.window",
            "model.mel_trsfm.init_args.window",
        )
        parser.link_arguments(
            "model.window",
            "model.lpc.init_args.window",
        )
        parser.link_arguments("model.table_size", "model.glottal.init_args.table_size")
        parser.link_arguments(
            "model.mel_trsfm.init_args.n_mels", "model.mel_model.init_args.in_channels"
        )
        # parser.link_arguments(
        #     "model.coarser_model_hidden_size", "model.table_model.init_args.in_channels"
        # )

        parser.link_arguments(
            (
                "model.voice_lpc_order",
                "model.noise_lpc_order",
                # "model.weighted_table",
                # "model.table_size",
                "model.coarser_model_hidden_size",
            ),
            "model.mel_model.init_args.out_channels",
            compute_fn=lambda voice_lpc_order, noise_lpc_order, coarser_model_hidden_size: voice_lpc_order
            + 1
            + noise_lpc_order
            + 1
            + coarser_model_hidden_size,
        )

        # parser.link_arguments(
        #     (
        #         "model.weighted_table",
        #         "model.table_size",
        #     ),
        #     "model.table_model.init_args.out_channels",
        #     compute_fn=lambda weighted_table, table_size: table_size
        #     if weighted_table
        #     else 1,
        # )

        parser.set_defaults(
            {
                "model.mel_model": {
                    "class_path": "models.mel.Mel2Control",
                },
                "model.glottal": {
                    "class_path": "models.synth.GlottalFlowTable",
                },
                "model.mel_trsfm": {
                    "class_path": "WrappedMelSpectrogram",
                },
            }
        )


def coeff_product(polynomials: Tensor) -> Tensor:
    n = polynomials.shape[0]
    if n == 1:
        return polynomials[0]

    c1 = coeff_product(polynomials[n // 2 :])
    c2 = coeff_product(polynomials[: n // 2])
    if c1.shape[1] > c2.shape[1]:
        c1, c2 = c2, c1
    weight = c1.unsqueeze(1).flip(2)
    prod = F.conv1d(
        c2.unsqueeze(0),
        weight,
        padding=weight.shape[2] - 1,
        groups=c2.shape[0],
    ).squeeze(0)
    return prod


def get_logits2biquads(
    rep_type: str,
    max_abs_pole: float = 0.99,
) -> Callable:
    if rep_type == "coef":

        def logits2coeff(logits: Tensor) -> Tensor:
            assert logits.shape[-1] == 2
            a1 = 2 * torch.tanh(logits[..., 0])
            a1_abs = a1.abs()
            a2 = 1 + 0.5 * a1_abs * (1 - torch.tanh(logits[..., 1]))
            return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)

    elif rep_type == "conj":

        def logits2coeff(logits: Tensor) -> Tensor:
            assert logits.shape[-1] == 2
            mag = torch.sigmoid(logits[..., 0]) * max_abs_pole
            phase = torch.sigmoid(logits[..., 1]) * torch.pi
            a1 = -2 * mag * torch.cos(phase)
            a2 = mag.square()
            return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)

    elif rep_type == "real":

        def logits2coeff(logits: Tensor) -> Tensor:
            assert logits.shape[-1] == 2
            z1 = torch.tanh(logits[..., 0]) * max_abs_pole
            z2 = torch.tanh(logits[..., 1]) * max_abs_pole
            a1 = -z1 - z2
            a2 = z1 * z2
            return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)

    else:
        raise ValueError(f"Unknown rep_type: {rep_type}, expected coef, conj or real")

    return logits2coeff


def get_biquads2lpc_coeffs(
    lpc_model: Union[BatchLPCSynth, BatchSecondOrderLPCSynth]
) -> Callable:
    if isinstance(lpc_model, BatchLPCSynth):
        return lambda biquads: coeff_product(
            biquads.view(-1, *biquads.shape[-2:]).transpose(0, 1)
        ).view(*biquads.shape[:2], -1)
    elif isinstance(lpc_model, BatchSecondOrderLPCSynth):
        return lambda biquads: biquads
    else:
        raise ValueError(
            f"Unknown lpc_model: {lpc_model}, expected BatchLPCSynth or BatchSecondOrderLPCSynth"
        )


def linear_upsample(x: Tensor, hop_length: int) -> Tensor:
    return F.interpolate(
        x.unsqueeze(1),
        (x.size(1) - 1) * hop_length + 1,
        mode="linear",
        align_corners=True,
    ).squeeze(1)


def smooth_phase_offset(phase_offset: Tensor) -> Tensor:
    # wrapp the differences into [-0.5, 0.5]
    return torch.cumsum(
        torch.cat(
            [phase_offset[:, :1], (phase_offset.diff(dim=1) + 0.5) % 1 - 0.5], dim=1
        ),
        dim=1,
    )


class MelGlottalVocoder(pl.LightningModule):
    def __init__(
        self,
        mel_model: Mel2Control,
        # table_model: Mel2Control,
        criterion: nn.Module,
        glottal: GlottalFlowTable,
        lpc: Union[BatchLPCSynth, BatchSecondOrderLPCSynth],
        mel_trsfm: WrappedMelSpectrogram,
        window: str = "hann",
        voice_lpc_order: int = 16,
        noise_lpc_order: int = 22,
        lpc_coeff_rep: str = "conj",
        sample_rate: int = 16000,
        hop_length: int = 80,
        table_size: int = 100,
        weighted_table: bool = False,
        coarser_hop_rate: int = 10,
        coarser_model_hidden_size: int = 128,
        max_abs_pole: float = 0.9,
        apply_radiation: bool = False,
        radiation_kernel_size: int = 256,
    ):
        super().__init__()

        self.model = mel_model
        self.criterion = criterion
        self.glottal = glottal
        self.lpc = lpc
        self.mel_trsfm = mel_trsfm

        self.register_buffer("log_mel_min", torch.tensor(torch.inf))
        self.register_buffer("log_mel_max", torch.tensor(-torch.inf))

        if apply_radiation:
            self.register_buffer(
                "radiation_filter",
                get_radiation_time_filter(
                    radiation_kernel_size, window_fn=get_window_fn(window)
                )
                .flip(0)
                .unsqueeze(0)
                .unsqueeze(0),
            )
            self.radiation_filter_padding = self.radiation_filter.shape[-1] // 2

        self.coarser_mel_model = nn.Sequential(
            nn.AvgPool1d(
                kernel_size=coarser_hop_rate,
                stride=coarser_hop_rate,
                padding=coarser_hop_rate // 2,
            ),
            nn.Conv1d(
                in_channels=coarser_model_hidden_size,
                out_channels=coarser_model_hidden_size * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                in_channels=coarser_model_hidden_size,
                out_channels=1 + (table_size if weighted_table else 1),  # plus offset
                kernel_size=1,
            ),
        )
        self.coarser_mel_model[-1].weight.data.zero_()
        self.coarser_mel_model[-1].bias.data.zero_()

        self.weighted_table = weighted_table
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.coarser_hop_length = hop_length * coarser_hop_rate
        self.max_abs_pole = max_abs_pole
        self.split_sizes = [
            voice_lpc_order,
            1,  # gain
            noise_lpc_order,  # noise coeffs
            1,  # noise gain
            coarser_model_hidden_size,
        ]
        self.coarser_split_sizes = [
            table_size if weighted_table else 1,
            1,  # offset
        ]
        self.logits2biquads = get_logits2biquads(lpc_coeff_rep, max_abs_pole)
        self.biquads2lpc_coeffs = get_biquads2lpc_coeffs(lpc)

        self.model.dense_out.bias.data[:voice_lpc_order] = (
            -10.0 if lpc_coeff_rep == "conj" else 0
        )  # voice coeffs
        self.model.dense_out.bias.data[voice_lpc_order] = 0.0  # gain
        self.model.dense_out.bias.data[
            voice_lpc_order + 1 : voice_lpc_order + 1 + noise_lpc_order
        ] = (
            -10.0 if lpc_coeff_rep == "conj" else 0
        )  # noise coeffs
        self.model.dense_out.bias.data[
            voice_lpc_order + 1 + noise_lpc_order
        ] = -5.0  # noise gain

    def x2log_mel(self, x):
        mel = self.mel_trsfm(x)
        log_mel = torch.log(mel + 1e-8)
        if self.training:
            self.log_mel_min.fill_(min(self.log_mel_min, log_mel.min().item()))
            self.log_mel_max.fill_(max(self.log_mel_max, log_mel.max().item()))
        return (log_mel - self.log_mel_min) / (self.log_mel_max - self.log_mel_min)

    def get_control_params(self, mel):
        control_params = self.model(mel)
        batch, frames, _ = control_params.shape
        (
            voice_coeffs_logits,
            voice_log_gain,
            noise_coeffs_logits,
            noise_log_gain,
            coarser_logits,
        ) = control_params.split(self.split_sizes, dim=2)
        voice_biquads = self.logits2biquads(
            voice_coeffs_logits.view(batch, frames, -1, 2)
        )
        noise_biquads = self.logits2biquads(
            noise_coeffs_logits.view(batch, frames, -1, 2)
        )

        return (
            voice_biquads,
            voice_log_gain.squeeze(2),
            noise_biquads,
            noise_log_gain.squeeze(2),
            coarser_logits,
        )

    def radiation(self, x):
        if hasattr(self, "radiation_filter"):
            return F.conv1d(
                x.unsqueeze(1),
                self.radiation_filter,
                padding=self.radiation_filter_padding,
            ).squeeze(1)
        return x

    def forward(self, x, f0_in_hz):
        instant_freq = f0_in_hz / self.sample_rate
        raw_phase = torch.cumsum(instant_freq, dim=1)

        (
            voice_biquads,
            voice_log_gain,
            noise_biquads,
            noise_log_gain,
            coarser_logits,
        ) = self.get_control_params(self.x2log_mel(x))

        voice_lpc_coeffs = self.biquads2lpc_coeffs(voice_biquads)
        noise_lpc_coeffs = self.biquads2lpc_coeffs(noise_biquads)

        table_control_logits, phase_offset_logits = (
            self.coarser_mel_model(coarser_logits.transpose(1, 2))
            .transpose(1, 2)
            .split(self.coarser_split_sizes, dim=2)
        )
        table_control = (
            table_control_logits.sigmoid().squeeze(2)
            if not self.weighted_table
            else table_control_logits.softmax(dim=2)
        )
        phase_offset = smooth_phase_offset(phase_offset_logits.squeeze(2).sigmoid())

        upsampled_phase_offset = linear_upsample(phase_offset, self.coarser_hop_length)
        phase = (
            raw_phase[:, : upsampled_phase_offset.shape[-1]]
            + upsampled_phase_offset[:, : raw_phase.shape[-1]]
        )
        glottal_flow = self.glottal(phase % 1, table_control, self.coarser_hop_length)
        noise = torch.randn_like(glottal_flow)

        vocal = self.lpc(
            glottal_flow, voice_log_gain.exp(), voice_lpc_coeffs
        ) + self.lpc(noise, noise_log_gain.exp(), noise_lpc_coeffs)

        return self.radiation(vocal)

    def training_step(self, batch, batch_idx):
        x, f0_in_hz = batch
        # x = x[:, 1, :]

        # mask = (f0_in_hz > 20).float()

        x_hat = self(x, f0_in_hz)

        loss = self.criterion(x_hat, x[..., : x_hat.shape[-1]])
        # time_loss = torch.mean(mask * (x_hat - x[..., : x_hat.shape[-1]]).abs())
        # loss = loss + time_loss
        # self.print(f"train_loss: {loss.item()}")
        self.log("train_loss", loss, prog_bar=False, sync_dist=True)
        # self.log("train_time_loss", time_loss, prog_bar=True, sync_dist=True)
        return loss
