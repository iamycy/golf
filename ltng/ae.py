import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Tuple, Callable, Union
from torchaudio.transforms import MelSpectrogram
import numpy as np
import yaml
import math
from importlib import import_module
from frechet_audio_distance import FrechetAudioDistance

from models.utils import get_window_fn
from models.mel import Mel2Control, WN
from models.utils import AudioTensor, get_f0, freq2cent
from models.ctrl import DUMMY_SPLIT_TRSFM
from models.synth import WrappedPhaseDownsampledIndexedGlottalFlowTable
from models.noise import NoiseInterface
from models.filters import SampleBasedLTVMinimumPhaseFilter, LTVZeroPhaseFIRFilter
from models.crepe import CREPE
from .vocoder import ScaledLogMelSpectrogram


class VoiceAutoEncoderCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "model.hop_length", "model.feature_trsfm.init_args.hop_length"
        )

        parser.link_arguments(
            "model.sample_rate", "model.feature_trsfm.init_args.sample_rate"
        )
        parser.link_arguments(
            "model.window",
            "model.feature_trsfm.init_args.window",
        )


class VoiceAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        feature_trsfm: ScaledLogMelSpectrogram,
        feature_encoder_type: str,
        feature_encoder_kwargs: Dict,
        # f0_encoder: nn.Module,
        harm_oscillator: WrappedPhaseDownsampledIndexedGlottalFlowTable,
        noise_generator: NoiseInterface,
        noise_filter: LTVZeroPhaseFIRFilter,
        end_filter: SampleBasedLTVMinimumPhaseFilter,
        criterion: nn.Module,
        window: str = "hanning",
        sample_rate: int = 24000,
        hop_length: int = 120,
        f0_min: float = 60,
        f0_max: float = 1000,
        l1_loss_weight: float = 0.0,
        f0_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.criterion = criterion
        self.feature_trsfm = feature_trsfm
        self.f0_encoder = CREPE(1, 513)
        self.offset_downsampler = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool1d(
                kernel_size=17,
                stride=8,
                padding=8,
            ),
            nn.Conv1d(
                in_channels=512,
                out_channels=256,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=256,
                out_channels=1,
                kernel_size=1,
            ),
        )

        self.harm_oscillator = harm_oscillator
        self.noise_generator = noise_generator
        self.noise_filter = noise_filter
        self.end_filter = end_filter

        ctrl_fns = [
            self.harm_oscillator.ctrl,
            self.noise_generator.ctrl,
            self.noise_filter.ctrl,
            self.end_filter.ctrl,
        ]
        split_trsfm = DUMMY_SPLIT_TRSFM
        for ctrl_fn in ctrl_fns[::-1]:
            split_trsfm = ctrl_fn(split_trsfm)
        self.split_tuples, self.trsfms = split_trsfm((), ())
        self._split_size = sum(self.split_tuples, ())

        module_path, class_name = feature_encoder_type.rsplit(".", 1)
        module = import_module(module_path)

        self.feature_encoder = getattr(module, class_name)(
            out_channels=sum(self._split_size), **feature_encoder_kwargs
        )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.l1_loss_weight = l1_loss_weight
        self.f0_loss_weight = f0_loss_weight

        self.log_f0_min = math.log(f0_min)
        self.log_f0_max = math.log(f0_max)

    def logits2f0(self, logits: AudioTensor) -> AudioTensor:
        return torch.exp(
            torch.sigmoid(logits) * (self.log_f0_max - self.log_f0_min)
            + self.log_f0_min
        )

    def forward(self, x: torch.Tensor):
        # wrapped_phase = self.f0_encoder(x.unsqueeze(1)).sigmoid().squeeze(1)

        # # get smooth wrapped_phase
        # pitch_mark = torch.diff(wrapped_phase, 1, 1) < -0.0
        # consecutive_pitch_mark = (pitch_mark & torch.roll(pitch_mark, 1, 1)) | (
        #     pitch_mark & torch.roll(pitch_mark, 2, 1)
        # )
        # pitch_mark[consecutive_pitch_mark] = False

        # batch_idx, time_idx = torch.nonzero(pitch_mark, as_tuple=True)
        # periods = torch.diff(time_idx)
        # valid_mask = torch.diff(batch_idx) == 0

        # mark_values = wrapped_phase[batch_idx, time_idx]
        # mark_values_diff = torch.diff(mark_values) + 1
        # phase_inc = mark_values_diff / periods

        # alter_value = []
        # for i in range(wrapped_phase.shape[0]):
        #     batch_mask = batch_idx == i
        #     mask_i = batch_mask[:-1] & valid_mask
        #     if not torch.any(mask_i):
        #         alter_value.append(wrapped_phase[i])
        #         continue
        #     expand_phase_inc = phase_inc[mask_i].repeat_interleave(periods[mask_i])
        #     smooth_phase = (
        #         torch.cumsum(expand_phase_inc, 0)
        #         + mark_values[torch.nonzero(mask_i)[0]]
        #     ) % 1

        #     alter_value_i = wrapped_phase.new_zeros(wrapped_phase.shape[1])
        #     time_idx_i = time_idx[batch_mask]
        #     alter_value_i[time_idx_i[0] + 1 : time_idx_i[-1]] = smooth_phase[:-1]
        #     alter_value.append(alter_value_i)

        # smooth_wrapped_phase = AudioTensor(torch.stack(alter_value, 0))

        logits = self.f0_encoder(x.unsqueeze(1))
        f0 = self.logits2f0(logits[..., 0])
        phase_offset = torch.sigmoid(
            AudioTensor(
                self.offset_downsampler(logits[..., 1:].as_tensor().mT).mT,
                hop_length=logits.hop_length * 8,
            )
        )
        freq = (
            (f0 / self.sample_rate)
            .reduce_hop_length()
            .unfold(phase_offset.hop_length, phase_offset.hop_length)
        )
        # freq._data = freq._data.detach()
        padded_freq = F.pad(freq, (1, 0), "constant", 0)
        wrapped_phase = (torch.cumsum(padded_freq, 2) + phase_offset) % 1
        wrapped_phase, pred_next_phase = wrapped_phase[..., :-1], wrapped_phase[..., -1]
        wrapped_phase = AudioTensor(
            wrapped_phase.as_tensor().reshape(wrapped_phase.shape[0], -1)
        )

        phase_match_loss = (
            torch.abs((pred_next_phase - phase_offset[:, 1:] + 0.5) % 1 - 0.5)
            .as_tensor()
            .mean()
        )

        feats = self.feature_trsfm(x)
        logits = self.feature_encoder(feats).split(self._split_size, dim=2)
        logits = [feats.new_tensor(torch.squeeze(l, 2)) for l in logits]

        groupped_logits = []
        for splits in self.split_tuples:
            groupped_logits.append(logits[: len(splits)])
            logits = logits[len(splits) :]

        harm_osc_params, noise_params, noise_filter_params, end_filter_params = map(
            lambda x: x[0](*x[1]), zip(self.trsfms, groupped_logits)
        )

        harm_osc = self.harm_oscillator(wrapped_phase, *harm_osc_params)
        src = harm_osc + self.noise_filter(
            self.noise_generator(harm_osc, *noise_params), *noise_filter_params
        )
        recon = self.end_filter(src, *end_filter_params)

        return phase_offset, f0, wrapped_phase, phase_match_loss, recon

    def f0_loss(self, f0_hat, f0):
        return F.l1_loss(torch.log(f0_hat + 1e-3), torch.log(f0 + 1e-3))

    def training_step(self, batch, batch_idx):
        x, f0_in_hz = batch
        f0_in_hz[f0_in_hz == 0] = 1
        mask = f0_in_hz > 50

        *_, f0_hat, wrapped_phase, phase_match_loss, x_hat = self(x)
        # f0_hat = (torch.diff(wrapped_phase, dim=1) + 1) % 1 * self.sample_rate
        x_hat = x_hat.as_tensor()
        low_res_f0 = f0_in_hz[:, :: f0_hat.hop_length]
        low_res_mask = mask[:, :: f0_hat.hop_length]
        f0_hat = f0_hat.as_tensor()[:, : low_res_f0.shape[-1]]
        low_res_f0 = low_res_f0[:, : f0_hat.shape[-1]]
        low_res_mask = low_res_mask[:, : f0_hat.shape[-1]]

        x = x[:, : x_hat.shape[-1]]
        mask = mask[:, : x_hat.shape[1]]
        x_hat = x_hat[:, : x.shape[1]]
        loss = self.criterion(x_hat, x) + phase_match_loss
        l1_loss = torch.sum(mask.float() * (x_hat - x).abs()) / mask.count_nonzero()

        # f0_in_hz = f0_in_hz[:, : f0_hat.shape[1]]
        f0_loss = self.f0_loss(f0_hat[low_res_mask], low_res_f0[low_res_mask])

        self.log(
            "train_phase_match_loss", phase_match_loss, prog_bar=True, sync_dist=True
        )
        self.log("train_l1_loss", l1_loss, prog_bar=True, sync_dist=True)
        self.log("train_f0_loss", f0_loss, prog_bar=True, sync_dist=True)
        if self.l1_loss_weight > 0:
            loss = loss + l1_loss * self.l1_loss_weight
        if self.f0_loss_weight > 0:
            loss = loss + f0_loss * self.f0_loss_weight

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.tmp_val_outputs = []

    def validation_step(self, batch, batch_idx):
        x, f0_in_hz = batch
        f0_in_hz[f0_in_hz == 0] = 1
        mask = f0_in_hz > 50

        *_, f0_hat, wrapped_phase, phase_match_loss, x_hat = self(x)
        # f0_hat = (torch.diff(wrapped_phase, dim=1) + 1) % 1 * self.sample_rate
        x_hat = x_hat.as_tensor()
        low_res_f0 = f0_in_hz[:, :: f0_hat.hop_length]
        low_res_mask = mask[:, :: f0_hat.hop_length]
        f0_hat = f0_hat.as_tensor()[:, : low_res_f0.shape[-1]]
        low_res_f0 = low_res_f0[:, : f0_hat.shape[-1]]
        low_res_mask = low_res_mask[:, : f0_hat.shape[-1]]

        x = x[:, : x_hat.shape[-1]]
        mask = mask[:, : x_hat.shape[1]]
        x_hat = x_hat[:, : x.shape[1]]
        loss = self.criterion(x_hat, x) + phase_match_loss
        l1_loss = torch.sum(mask.float() * (x_hat - x).abs()) / mask.count_nonzero()

        # f0_in_hz = f0_in_hz[:, : f0_hat.shape[1]]
        f0_loss = self.f0_loss(f0_hat[low_res_mask], low_res_f0[low_res_mask])

        if self.l1_loss_weight > 0:
            loss = loss + l1_loss * self.l1_loss_weight
        if self.f0_loss_weight > 0:
            loss = loss + f0_loss * self.f0_loss_weight

        self.tmp_val_outputs.append((loss, l1_loss, f0_loss, phase_match_loss))

        return loss

    def on_validation_epoch_end(self) -> None:
        outputs = self.tmp_val_outputs
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_l1_loss = sum(x[1] for x in outputs) / len(outputs)
        avg_f0_loss = sum(x[2] for x in outputs) / len(outputs)
        avg_phase_match_loss = sum(x[3] for x in outputs) / len(outputs)

        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_l1_loss", avg_l1_loss, prog_bar=False, sync_dist=True)
        self.log("val_f0_loss", avg_f0_loss, prog_bar=False, sync_dist=True)
        self.log(
            "val_phase_match_loss",
            avg_phase_match_loss,
            prog_bar=False,
            sync_dist=True,
        )

        delattr(self, "tmp_val_outputs")
