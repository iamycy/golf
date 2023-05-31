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
from torchaudio.transforms import Preemphasis, Deemphasis

from models.utils import get_window_fn, rc2lpc, fir_filt, AudioTensor
from lpcnet import SampleNet, ContinuousMuLawDecoding, ContinuousMuLawEncoding


class LPCNetVocoderCLI(LightningCLI):
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

        parser.link_arguments(
            "model.quantization_channels",
            "model.sample_decoder.init_args.quantization_channels",
        )


class LPCNetVocoder(pl.LightningModule):
    def __init__(
        self,
        feature_trsfm: torch.nn.Module,
        frame_decoder: torch.nn.Module,
        sample_decoder: SampleNet,
        lpc_order: int = 22,
        quantization_channels: int = 256,
        alpha: float = 0.85,
        window: str = "hanning",
        sample_rate: int = 24000,
        hop_length: int = 120,
        gamma: float = 1.0,
    ):
        super().__init__()

        # self.save_hyperparameters()

        self.frame_decoder = nn.Sequential(frame_decoder, nn.Tanh())
        self.sample_decoder = sample_decoder
        self.feature_trsfm = feature_trsfm
        self.mu_enc = ContinuousMuLawEncoding(quantization_channels)
        self.mu_dec = ContinuousMuLawDecoding(quantization_channels)
        self.pre_emphasis = Preemphasis(alpha)
        self.de_emphasis = Deemphasis(alpha)

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.lpc_order = lpc_order
        self.gamma = gamma

        mu = quantization_channels - 1.0
        self.regularizer = (
            lambda x: (x - 0.5 * mu).abs().mean() * math.log1p(mu) / mu * 2
        )

    def forward(self, feats: torch.Tensor):
        (f0, *other_params, voicing_logits) = self.encoder(feats)

        phase = f0 / self.sample_rate
        if voicing_logits is not None:
            voicing = torch.sigmoid(voicing_logits)
        else:
            voicing = None

        return (
            f0,
            self.decoder(phase, *other_params, voicing=voicing),
            voicing,
        )

    def on_train_start(self) -> None:
        if self.logger is not None:
            self.logger.watch(
                self.frame_decoder, log_freq=1000, log="all", log_graph=False
            )
            self.logger.watch(
                self.sample_decoder, log_freq=1000, log="all", log_graph=False
            )

    def on_train_end(self) -> None:
        if self.logger is not None:
            self.logger.experiment.unwatch(self.frame_decoder)
            self.logger.experiment.unwatch(self.sample_decoder)

    def interp_loss(self, e_mu, logits):
        q = logits.shape[1]
        e_mu = e_mu.unsqueeze(1)
        lower_idx = torch.floor(e_mu).long().clip(0, q - 2)
        upper_idx = lower_idx + 1
        p = e_mu - lower_idx
        log_prob = F.log_softmax(logits, dim=1)
        selected_log_probs = (
            torch.gather(log_prob, 1, lower_idx) * (1 - p)
            + torch.gather(log_prob, 1, upper_idx) * p
        )
        return selected_log_probs.mean(), self.regularizer(e_mu)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        s = self.pre_emphasis(x)
        # s = torch.clip(s, -1, 1)
        assert ~torch.any(torch.isnan(s)), "NaN in input signal"
        feats = self.feature_trsfm(x)

        f = self.frame_decoder(feats)
        lpc = rc2lpc(f[..., : self.lpc_order].as_tensor())

        f = f.reduce_hop_length().as_tensor()
        upsampled_lpc = (
            AudioTensor(lpc, hop_length=self.hop_length).reduce_hop_length().as_tensor()
        )
        minimum_length = min(upsampled_lpc.shape[1], s.shape[1])
        s = s[:, :minimum_length]
        upsampled_lpc = upsampled_lpc[:, :minimum_length]
        f = f[:, :minimum_length]
        weight = torch.cat([torch.ones_like(upsampled_lpc[..., :1]), upsampled_lpc], 2)
        e = fir_filt(s, weight)
        assert ~torch.any(torch.isnan(e)), "NaN in excitation signal"
        p = s - e

        p_mu = self.mu_enc(p.clip(-1, 1))
        e_mu = self.mu_enc(e.clip(-1, 1))
        s_mu = self.mu_enc(s.clip(-1, 1))

        pred_e_logits = self.sample_decoder(
            f[:, 1:], p_mu[:, 1:], s_mu[:, :-1], e_mu[:, :-1]
        )

        ll, reg = self.interp_loss(e_mu[:, 1:], pred_e_logits.transpose(1, 2))
        loss = -ll + self.gamma * reg

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_ll", ll, prog_bar=False, sync_dist=True)
        self.log("train_reg", reg, prog_bar=False, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.tmp_val_outputs = []

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        s = self.pre_emphasis(x)
        feats = self.feature_trsfm(x)

        f = self.frame_decoder(feats)
        lpc = rc2lpc(f[..., : self.lpc_order].as_tensor())

        f = f.reduce_hop_length().as_tensor()
        upsampled_lpc = (
            AudioTensor(lpc, hop_length=self.hop_length).reduce_hop_length().as_tensor()
        )
        minimum_length = min(upsampled_lpc.shape[1], s.shape[1])
        s = s[:, :minimum_length]
        upsampled_lpc = upsampled_lpc[:, :minimum_length]
        f = f[:, :minimum_length]
        weight = torch.cat([torch.ones_like(upsampled_lpc[..., :1]), upsampled_lpc], 2)
        e = fir_filt(s, weight)
        p = s - e

        p_mu = self.mu_enc(p)
        e_mu = self.mu_enc(e)
        s_mu = self.mu_enc(s)

        pred_e_logits = self.sample_decoder(
            f[:, 1:], p_mu[:, 1:], s_mu[:, :-1], e_mu[:, :-1]
        )

        ll, reg = self.interp_loss(e_mu[:, 1:], pred_e_logits.transpose(1, 2))
        loss = -ll + self.gamma * reg

        self.tmp_val_outputs.append((loss, ll, reg))

        return loss

    def on_validation_epoch_end(self) -> None:
        outputs = self.tmp_val_outputs
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_ll = sum(x[1] for x in outputs) / len(outputs)
        avg_reg = sum(x[2] for x in outputs) / len(outputs)

        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_ll", avg_ll, prog_bar=False, sync_dist=True)
        self.log("val_reg", avg_reg, prog_bar=False, sync_dist=True)

        delattr(self, "tmp_val_outputs")
