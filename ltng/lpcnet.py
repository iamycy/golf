import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Tuple, Callable, Union
from torchaudio.transforms import MelSpectrogram
import numpy as np
import yaml
from tqdm import tqdm
import math
from importlib import import_module
from frechet_audio_distance import FrechetAudioDistance
from torchaudio.transforms import Preemphasis, Deemphasis
from diffsptk import (
    Frame,
    Window,
    LPC,
    LogAreaRatioToParcorCoefficients,
    ParcorCoefficientsToLinearPredictiveCoefficients,
    ParcorCoefficientsToLogAreaRatio,
    LinearPredictiveCoefficientsToParcorCoefficients,
)

from models.utils import get_window_fn, rc2lpc, fir_filt, AudioTensor, get_f0, freq2cent
from models.lpcnet import SampleNet, ContinuousMuLawDecoding, ContinuousMuLawEncoding


class Clip(nn.Module):
    def __init__(self, min: float, max: float):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return x.clip(self.min, self.max)


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
        match_lpc: bool = False,
        lpc_frame_lengeth: int = 1024,
    ):
        super().__init__()

        # self.save_hyperparameters()

        self.frame_decoder = frame_decoder
        self.sample_decoder = sample_decoder
        self.feature_trsfm = feature_trsfm
        self.mu_enc = ContinuousMuLawEncoding(quantization_channels)
        self.mu_dec = ContinuousMuLawDecoding(quantization_channels)
        self.pre_emphasis = Preemphasis(alpha)
        self.de_emphasis = Deemphasis(alpha)
        self.lar2lpc = nn.Sequential(
            LogAreaRatioToParcorCoefficients(lpc_order),
            ParcorCoefficientsToLinearPredictiveCoefficients(lpc_order),
        )

        self.x2lar = nn.Sequential(
            Frame(lpc_frame_lengeth, hop_length),
            Window(lpc_frame_lengeth, window=window),
            LPC(lpc_order, lpc_frame_lengeth),
            LinearPredictiveCoefficientsToParcorCoefficients(
                lpc_order, warn_type="warn"
            ),
            Clip(-0.999999, 0.999999),
            ParcorCoefficientsToLogAreaRatio(lpc_order),
        )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.lpc_order = lpc_order
        self.gamma = gamma
        self.match_lpc = match_lpc

        mu = quantization_channels - 1.0
        self.regularizer = (
            lambda x: (x - 0.5 * mu).abs().mean() * math.log1p(mu) / mu * 2
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
        p = p.clip(0, 1)
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
        lar = f[..., : self.lpc_order].as_tensor() * 2
        lpc = self.lar2lpc(torch.cat([torch.zeros_like(lar[..., :1]), lar], 2))[..., 1:]
        # lpc = rc2lpc(f[..., : self.lpc_order].as_tensor())

        f = f.reduce_hop_length().as_tensor().tanh()
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

        p_mu = self.mu_enc(p)
        e_mu = self.mu_enc(e)
        s_mu = self.mu_enc(s)

        pred_e_logits = self.sample_decoder(
            f[:, 1:], p_mu[:, 1:], s_mu[:, :-1], e_mu[:, :-1]
        )

        ll, reg = self.interp_loss(e_mu[:, 1:], pred_e_logits.transpose(1, 2))
        loss = -ll + self.gamma * reg

        if self.match_lpc:
            with torch.cuda.amp.autocast(enabled=False):
                gt_lar = self.x2lar(x.add(1e-7))[..., 1:].to(x.dtype)
                assert ~torch.any(torch.isnan(gt_lar)), "NaN in ground truth LAR"
                assert ~torch.any(torch.isinf(gt_lar)), "Inf in ground truth LAR"
            lar_l2 = F.mse_loss(lar[:, : gt_lar.shape[1]], gt_lar)
            loss += lar_l2
            self.log("train_lar_l2", lar_l2, prog_bar=False, sync_dist=True)

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
        lar = f[..., : self.lpc_order].as_tensor() * 2
        lpc = self.lar2lpc(torch.cat([torch.zeros_like(lar[..., :1]), lar], 2))[..., 1:]

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

        outputs = (loss, ll, reg)

        if self.match_lpc:
            with torch.cuda.amp.autocast(enabled=False):
                gt_lar = self.x2lar(x.add(1e-7))[..., 1:].to(x.dtype)
            lar_l2 = F.mse_loss(lar[:, : gt_lar.shape[1]], gt_lar)
            loss += lar_l2
            outputs += (lar_l2,)

        self.tmp_val_outputs.append(outputs)

        return loss

    def on_validation_epoch_end(self) -> None:
        outputs = self.tmp_val_outputs
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_ll = sum(x[1] for x in outputs) / len(outputs)
        avg_reg = sum(x[2] for x in outputs) / len(outputs)

        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_ll", avg_ll, prog_bar=False, sync_dist=True)
        self.log("val_reg", avg_reg, prog_bar=False, sync_dist=True)

        if self.match_lpc:
            avg_lar_l2 = sum(x[3] for x in outputs) / len(outputs)
            self.log("val_lar_l2", avg_lar_l2, prog_bar=False, sync_dist=True)

        delattr(self, "tmp_val_outputs")

    def on_test_start(self) -> None:
        frechet = FrechetAudioDistance(
            use_pca=False, use_activation=False, verbose=True
        )
        frechet.model = frechet.model.to(self.device)
        self.frechet = frechet

        self.tmp_test_outputs = []

        return super().on_test_start()

    def test_step(self, batch, batch_idx):
        x, f0_in_hz = batch
        f0_in_hz = f0_in_hz[:, :: self.hop_length].cpu().numpy()

        s = self.pre_emphasis(x)
        feats = self.feature_trsfm(x)

        f = self.frame_decoder(feats)
        lpc = rc2lpc(f[..., : self.lpc_order].as_tensor())

        f = f.reduce_hop_length().as_tensor()
        upsampled_lpc = (
            AudioTensor(lpc, hop_length=self.hop_length).reduce_hop_length().as_tensor()
        )
        minimum_length = min(upsampled_lpc.shape[1], s.shape[1])
        lpc_order = upsampled_lpc.shape[2]
        s = s[:, :minimum_length]
        upsampled_lpc = upsampled_lpc[:, :minimum_length].flip(2)
        f = f[:, :minimum_length]

        s_buffer = f.new_zeros(f.shape[0], lpc_order + minimum_length)
        e_mu_prev = (
            f.new_ones(f.shape[0]) * (self.mu_enc.quantization_channels - 1) * 0.5
        )
        states = None
        for i in tqdm(range(minimum_length)):
            p = -torch.sum(s_buffer[:, i : i + lpc_order] * upsampled_lpc[:, i], dim=1)
            logits, states = self.sample_decoder.sample_forward(
                f[:, i],
                self.mu_enc(p),
                self.mu_enc(s_buffer[:, i + lpc_order - 1]),
                e_mu_prev,
                states,
            )
            probs = F.softmax(logits * 2, dim=1)
            dist = torch.distributions.Categorical(probs=probs)
            e_mu = dist.sample().float()
            e = self.mu_dec(e_mu)
            pred = e + p
            s_buffer[:, i + lpc_order] = pred.clip(-1, 1)
            e_mu_prev = e_mu

        s_hat = s_buffer[:, lpc_order:]
        x_hat = self.de_emphasis(s_hat)

        x_hat = x_hat.cpu().numpy().astype(np.float64)
        x = x.cpu().numpy().astype(np.float64)
        N = x_hat.shape[0]
        f0_hat_list = []
        x_true_embs = []
        x_hat_embs = []
        for i in range(N):
            f0_hat, _ = get_f0(x_hat[i], self.sample_rate)
            f0_hat_list.append(f0_hat)

            x_hat_emb = (
                self.frechet.model.forward(x_hat[i], self.sample_rate).cpu().numpy()
            )
            x_hat_embs.append(x_hat_emb)
            x_emb = self.frechet.model.forward(x[i], self.sample_rate).cpu().numpy()
            x_true_embs.append(x_emb)

        x_true_embs = np.concatenate(x_true_embs, axis=0)
        x_hat_embs = np.concatenate(x_hat_embs, axis=0)
        f0_hat = np.stack(f0_hat_list, axis=0)
        f0_in_hz = f0_in_hz[:, : f0_hat.shape[1]]
        f0_hat = f0_hat[:, : f0_in_hz.shape[1]]
        f0_in_hz = np.maximum(f0_in_hz, 80)
        f0_hat = np.maximum(f0_hat, 80)
        f0_loss = np.mean(np.abs(freq2cent(f0_hat) - freq2cent(f0_in_hz)))

        self.tmp_test_outputs.append((f0_loss, x_true_embs, x_hat_embs, N))

        return f0_loss, x_true_embs, x_hat_embs, N

    def on_test_epoch_end(self) -> None:
        outputs = self.tmp_test_outputs
        weights = [x[3] for x in outputs]
        avg_f0_loss = np.average([x[0] for x in outputs], weights=weights)

        x_true_embs = np.concatenate([x[1] for x in outputs], axis=0)
        x_hat_embs = np.concatenate([x[2] for x in outputs], axis=0)

        mu_background, sigma_background = self.frechet.calculate_embd_statistics(
            x_hat_embs
        )
        mu_eval, sigma_eval = self.frechet.calculate_embd_statistics(x_true_embs)
        fad_score = self.frechet.calculate_frechet_distance(
            mu_background, sigma_background, mu_eval, sigma_eval
        )

        self.log_dict(
            {
                "avg_f0_loss": avg_f0_loss,
                "fad_score": fad_score,
            },
            prog_bar=True,
            sync_dist=True,
        )
        delattr(self, "tmp_test_outputs")
        return
