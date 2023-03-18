import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Tuple, Callable, Union
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import lfilter, filtfilt
import diffsptk
import random
from functools import reduce

from models.synth import GlottalSynth, GlottalFlowTable
from models.lpc import LPCSynth, BatchLPCSynth, BatchSecondOrderLPCSynth
from models.mel import Mel2Control
from models.tspn import TTSPNEncoder, TopNGenerator
from models.utils import get_window_fn, get_radiation_time_filter, fir_filt


class WrappedMelSpectrogram(MelSpectrogram):
    def __init__(self, window: str, **kwargs):
        super().__init__(window_fn=get_window_fn(window), **kwargs)


class MelVocoderCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "model.hop_length", "model.mel_trsfm.init_args.hop_length"
        )
        parser.link_arguments("model.hop_length", "model.lpc.init_args.hop_length")

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
            (
                "model.mel_trsfm.init_args.n_mels",
                "model.enc_lpc_order",
                "model.enc_channels",
                "model.enc_stride",
            ),
            "model.mel_model.init_args.in_channels",
            compute_fn=lambda n_mels, enc_lpc_order, enc_channels, enc_stride: n_mels
            + enc_lpc_order
            + 1
            + enc_channels * 2 ** (len(enc_stride) - 1),
        )

        parser.link_arguments(
            (
                "model.ttspn_key_size",
                "model.ttspn_value_size",
            ),
            "model.voice_ttspn.init_args.d_model",
            compute_fn=lambda ttspn_key_size, ttspn_value_size: ttspn_key_size
            + ttspn_value_size,
        )
        parser.link_arguments(
            (
                "model.ttspn_key_size",
                "model.ttspn_value_size",
            ),
            "model.noise_ttspn.init_args.d_model",
            compute_fn=lambda ttspn_key_size, ttspn_value_size: ttspn_key_size
            + ttspn_value_size,
        )

        parser.link_arguments(
            (
                # "model.voice_lpc_order",
                # "model.noise_lpc_order",
                "model.ttspn_key_size",
                "model.voice_allpass_filter_order",
                # "model.weighted_table",
                # "model.table_size",
                "model.coarser_model_hidden_size",
            ),
            "model.mel_model.init_args.out_channels",
            compute_fn=lambda ttspn_key_size, voice_allpass_filter_order, coarser_model_hidden_size: ttspn_key_size
            * 2
            + 1
            + 1
            + voice_allpass_filter_order
            + coarser_model_hidden_size,
        )

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
                "model.voice_ttspn": {
                    "class_path": "models.tspn.TTSPNEncoder",
                    "init_args": {
                        "out_channels": 2,
                    },
                },
                "model.noise_ttspn": {
                    "class_path": "models.tspn.TTSPNEncoder",
                    "init_args": {
                        "out_channels": 2,
                    },
                },
            }
        )


def coeff_product(polynomials: Union[Tensor, List[Tensor]]) -> Tensor:
    n = len(polynomials)
    if n == 1:
        return polynomials[0]

    c1 = coeff_product(polynomials[n // 2 :])
    c2 = coeff_product(polynomials[: n // 2])
    if c1.shape[1] > c2.shape[1]:
        c1, c2 = c2, c1
    # outer = c1.unsqueeze(2) * c2.unsqueeze(1)
    # prod = (
    #     F.pad(outer, (0, c1.shape[1]))
    #     .view(c1.shape[0], -1, c1.shape[1])[:, :-1]
    #     .view(c1.shape[0], c1.shape[1], -1)
    #     .sum(1)
    # )
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
            a1 = 2 * torch.tanh(logits[..., 0]) * max_abs_pole
            a1_abs = a1.abs()
            a2 = 0.5 * (
                (2 - a1_abs) * torch.tanh(logits[..., 1]) * max_abs_pole + a1_abs
            )
            return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)

    elif rep_type == "conj":

        def logits2coeff(logits: Tensor) -> Tensor:
            assert logits.shape[-1] == 2
            mag = torch.sigmoid(logits[..., 0]) * max_abs_pole
            # phase = torch.sigmoid(logits[..., 1]) * torch.pi
            cos = torch.tanh(logits[..., 1])
            a1 = -2 * mag * cos
            a2 = mag.square()
            # real = torch.tanh(logits[..., 0]) * max_abs_pole
            # imag = torch.tanh(logits[..., 1]) * max_abs_pole
            # a1 = -2 * real
            # a2 = real.square() + imag.square()
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
        ).view(*biquads.shape[:2], -1)[..., 1:]
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
        voice_ttspn: TTSPNEncoder,
        noise_ttspn: TTSPNEncoder,
        window: str = "hanning",
        voice_lpc_order: int = 16,
        noise_lpc_order: int = 22,
        voice_allpass_filter_order: int = 6,
        ttspn_key_size: int = 32,
        ttspn_value_size: int = 64,
        ref_set_size: int = 128,
        enc_lpc_order: int = 50,
        enc_channels: int = 32,
        enc_stride: List[int] = [5, 2, 2, 2, 2],
        lpc_coeff_rep: str = "conj",
        src_smooth_alpha: float = 0.5,
        sample_rate: int = 16000,
        hop_length: int = 80,
        table_size: int = 100,
        weighted_table: bool = False,
        coarser_hop_rate: int = 10,
        coarser_model_hidden_size: int = 128,
        max_abs_pole: float = 0.9,
        apply_radiation: bool = False,
        radiation_kernel_size: int = 256,
        allpass_filter_order: int = 0,
        l1_loss_weight: float = 0.0,
    ):
        super().__init__()

        assert hop_length == reduce(
            lambda x, y: x * y, enc_stride
        ), f"Sum of strides ({sum(enc_stride)}) must be equal to hop_length ({hop_length})"

        self.model = mel_model
        self.criterion = criterion
        self.glottal = glottal
        self.lpc = lpc
        self.mel_trsfm = mel_trsfm
        self.voice_ttspn = voice_ttspn
        self.noise_ttspn = noise_ttspn

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

        modules = []
        in_channels = 2
        for stride in enc_stride:
            modules += [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=enc_channels,
                    kernel_size=stride * 2 + 1,
                    stride=stride,
                    padding=stride,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=enc_channels,
                    out_channels=enc_channels * 2,
                    kernel_size=1,
                ),
                nn.GLU(dim=1),
            ]
            in_channels = enc_channels
            enc_channels *= 2

        self.src_encoder = nn.Sequential(*modules)
        self.enc_lpc = nn.Sequential(
            diffsptk.Frame(hop_length * 4, hop_length),
            diffsptk.Window(hop_length * 4, window=window),
            diffsptk.LPC(enc_lpc_order, hop_length * 4),
        )

        self.coarser_mel_model[-1].weight.data.zero_()
        self.coarser_mel_model[-1].bias.data.zero_()

        self.weighted_table = weighted_table
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.coarser_hop_length = hop_length * coarser_hop_rate
        self.max_abs_pole = max_abs_pole
        self.l1_loss_weight = l1_loss_weight
        self.split_sizes = [
            ttspn_key_size,
            1,  # gain
            ttspn_key_size,  # noise coeffs
            1,  # noise gain
            coarser_model_hidden_size,
        ]
        self.voice_lpc_order = voice_lpc_order
        self.noise_lpc_order = noise_lpc_order
        self.voice_allpass_filter_order = voice_allpass_filter_order

        if voice_allpass_filter_order > 0:
            self.split_sizes.append(voice_allpass_filter_order)

        self.coarser_split_sizes = [
            table_size if weighted_table else 1,
            1,  # offset
        ]
        self.logits2biquads = get_logits2biquads(lpc_coeff_rep, max_abs_pole)
        self.biquads2lpc_coeffs = get_biquads2lpc_coeffs(lpc)

        # self.model.dense_out.weight.data.zero_()
        # self.model.dense_out.bias.data.uniform_(-5, 5)
        self.voice_ttspn.linear.weight.data.zero_()
        self.noise_ttspn.linear.weight.data.zero_()
        if lpc_coeff_rep == "conj":
            # self.model.dense_out.bias.data[:voice_lpc_order:2] = -10
            # self.model.dense_out.bias.data[
            #     voice_lpc_order + 1 : voice_lpc_order + 1 + noise_lpc_order : 2
            # ] = -10
            self.voice_ttspn.linear.bias.data[0] = -10
            self.noise_ttspn.linear.bias.data[0] = -10
        elif lpc_coeff_rep == "real" or lpc_coeff_rep == "coef":
            # self.model.dense_out.bias.data[:voice_lpc_order] = 0
            # self.model.dense_out.bias.data[
            #     voice_lpc_order + 1 : voice_lpc_order + 1 + noise_lpc_order
            # ] = 0
            self.voice_ttspn.linear.bias.data.zero_()
            self.noise_ttspn.linear.bias.data.zero_()

        self.model.dense_out.bias.data[ttspn_key_size] = 0.0  # gain
        self.model.dense_out.bias.data[
            ttspn_key_size + 1 + ttspn_key_size
        ] = -10.0  # noise gain

        self.register_buffer("src_smooth_a", torch.tensor([1, -src_smooth_alpha]))
        self.register_buffer("src_smooth_b", torch.tensor([1.0, 0.0]))

        if allpass_filter_order > 0:
            init = torch.randn(allpass_filter_order // 2, 2)
            # if lpc_coeff_rep == "conj":
            #     init[:, 0] = -10.0
            # elif lpc_coeff_rep == "real" or lpc_coeff_rep == "coef":
            #     init[:, 0] = 0.0
            self.register_parameter(
                "allpass_filter_logits",
                nn.Parameter(init),
            )

        self.topn = TopNGenerator(
            num_emb=ref_set_size,
            key_emb_size=ttspn_key_size,
            value_emb_size=ttspn_value_size,
        )

    def enc_src_and_lpc(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            lpc = self.enc_lpc(x.double().add(1e-7))
        lpc = torch.nan_to_num(lpc, nan=0.0, posinf=0.0, neginf=0.0)
        lpc = lpc.to(x.dtype)
        batch, frames, order = lpc.shape
        fir_weight = (
            linear_upsample(lpc.transpose(1, 2).reshape(-1, frames), self.hop_length)
            .view(batch, order, -1)
            .transpose(1, 2)
        )
        gain = fir_weight[..., 0] + 1e-7
        fir_weight[..., 0] = 1.0
        src = fir_filt(x[:, : fir_weight.shape[1]], fir_weight) / gain
        # smooth src, forward and backward
        src.relu_()
        src = filtfilt(src, self.src_smooth_a, self.src_smooth_b, clamp=False)
        return src, lpc

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
            # voice_coeffs_logits,
            voice_coeff_embed,
            voice_log_gain,
            # noise_coeffs_logits,
            noise_coeff_embed,
            noise_log_gain,
            coarser_logits,
            *_,
        ) = control_params.split(self.split_sizes, dim=2)

        voice_init_set = self.topn(
            voice_coeff_embed.view(-1, voice_coeff_embed.shape[-1]),
            self.voice_lpc_order,
        )
        voice_init_set = voice_init_set.view(batch, frames, self.voice_lpc_order, -1)
        noise_init_set = self.topn(
            noise_coeff_embed.view(-1, noise_coeff_embed.shape[-1]),
            self.noise_lpc_order,
        )
        noise_init_set = noise_init_set.view(batch, frames, self.noise_lpc_order, -1)

        voice_init_set = torch.cat(
            [
                voice_init_set,
                voice_coeff_embed.unsqueeze(-2).repeat(1, 1, self.voice_lpc_order, 1),
            ],
            dim=-1,
        )
        noise_init_set = torch.cat(
            [
                noise_init_set,
                noise_coeff_embed.unsqueeze(-2).repeat(1, 1, self.noise_lpc_order, 1),
            ],
            dim=-1,
        )
        voice_coeffs_logits = self.voice_ttspn(voice_init_set)
        noise_coeffs_logits = self.noise_ttspn(noise_init_set)

        voice_biquads = self.logits2biquads(
            voice_coeffs_logits  # .view(batch, frames, -1, 2)
        )
        noise_biquads = self.logits2biquads(
            noise_coeffs_logits  # .view(batch, frames, -1, 2)
        )

        if len(_) > 0:
            voice_allpass_biquads = self.logits2biquads(_[0].view(batch, frames, -1, 2))
            return (
                voice_biquads,
                voice_log_gain.squeeze(2),
                noise_biquads,
                noise_log_gain.squeeze(2),
                coarser_logits,
                voice_allpass_biquads,
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

        src, lpc = self.enc_src_and_lpc(x)
        lpc = lpc.transpose(1, 2)
        h = self.src_encoder(
            torch.stack([src, raw_phase[:, : src.shape[1]] % 1], dim=1)
        )
        mel = self.x2log_mel(x)
        frames = min(h.shape[2], lpc.shape[2], mel.shape[2])
        feats = torch.cat(
            [h[:, :, :frames], lpc[:, :, :frames], mel[:, :, :frames]], dim=1
        )
        # feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        # assert torch.all(torch.isfinite(feats))

        (
            voice_biquads,
            voice_log_gain,
            noise_biquads,
            noise_log_gain,
            coarser_logits,
            *_,
        ) = self.get_control_params(feats)

        voice_lpc_coeffs = self.biquads2lpc_coeffs(voice_biquads)
        noise_lpc_coeffs = self.biquads2lpc_coeffs(noise_biquads)
        if len(_) > 0:
            voice_allpass_biquads = _[0]
            voice_allpass_lpc_coeffs = self.biquads2lpc_coeffs(voice_allpass_biquads)
            voice_allpass_a = F.pad(voice_allpass_lpc_coeffs, (1, 0), value=1.0)
            voice_lpc_coeffs = F.pad(voice_lpc_coeffs, (1, 0), value=1.0)
            voice_allpass_b = voice_allpass_a.flip(-1)

            voice_lpc_coeffs = coeff_product(
                [
                    voice_lpc_coeffs.view(-1, voice_lpc_coeffs.shape[-1]),
                    voice_allpass_a.view(-1, voice_allpass_a.shape[-1]),
                ]
            ).view(*voice_lpc_coeffs.shape[:2], -1)[..., 1:]
        else:
            voice_allpass_b = None

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

        vocal = self.radiation(
            self.lpc(glottal_flow, voice_log_gain.exp(), voice_lpc_coeffs)
            + self.lpc(noise, noise_log_gain.exp(), noise_lpc_coeffs)
        )

        if voice_allpass_b is not None:
            fir_weights = (
                linear_upsample(
                    voice_allpass_b.transpose(1, 2).reshape(
                        -1, voice_allpass_b.shape[1]
                    ),
                    self.hop_length,
                )
                .view(voice_allpass_b.shape[0], voice_allpass_b.shape[2], -1)
                .transpose(1, 2)
            )
            vocal = fir_filt(
                vocal[:, : fir_weights.shape[1]], fir_weights[:, : vocal.shape[1]]
            )

        if hasattr(self, "allpass_filter_logits"):
            biquads = self.logits2biquads(self.allpass_filter_logits)
            allpass_a = coeff_product(biquads.unsqueeze(1)).squeeze()
            allpass_b = allpass_a.flip(0)
            vocal = lfilter(vocal, allpass_a, allpass_b, clamp=False)

        return vocal

    def training_step(self, batch, batch_idx):
        x, f0_in_hz = batch
        # x = x[:, 1, :]

        mask = f0_in_hz > 20
        num_nonzero = mask.count_nonzero()

        x_hat = self(x, f0_in_hz)

        loss = self.criterion(x_hat, x[..., : x_hat.shape[-1]])
        # l1_loss = F.l1_loss(x_hat, x[..., : x_hat.shape[-1]])

        l1_loss = (
            torch.sum(
                mask.float()[:, : x_hat.shape[1]]
                * (x_hat - x[..., : x_hat.shape[-1]]).abs()
            )
            / num_nonzero
        )

        self.log("train_l1_loss", l1_loss, prog_bar=False, sync_dist=True)
        loss = loss + l1_loss * self.l1_loss_weight
        # loss = loss + time_loss
        # self.print(f"train_loss: {loss.item()}")
        self.log("train_loss", loss, prog_bar=False, sync_dist=True)
        # self.log("train_time_loss", time_loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, f0_in_hz = batch

        x_hat = self(x, f0_in_hz)

        loss = self.criterion(x_hat, x[..., : x_hat.shape[-1]])
        l1_loss = F.l1_loss(x_hat, x[..., : x_hat.shape[-1]])
        loss = loss + l1_loss * self.l1_loss_weight
        return loss, l1_loss

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_l1_loss = sum(x[1] for x in outputs) / len(outputs)

        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_l1_loss", avg_l1_loss, prog_bar=False, sync_dist=True)
