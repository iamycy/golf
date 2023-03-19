import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Tuple, Callable, Union
from torchaudio.transforms import MelSpectrogram

from models.utils import get_window_fn
from models.hpn import HarmonicPlusNoiseSynth
from models.enc import VocoderParameterEncoderInterface
from models.utils import TimeContext


class ScaledLogMelSpectrogram(MelSpectrogram):
    def __init__(self, window: str, **kwargs):
        super().__init__(window_fn=get_window_fn(window), **kwargs)

        self.register_buffer("log_mel_min", torch.tensor(torch.inf))
        self.register_buffer("log_mel_max", torch.tensor(-torch.inf))

    def forward(self, waveform: Tensor) -> Tensor:
        mel = super().forward(waveform).transpose(-1, -2)
        log_mel = torch.log(mel + 1e-8)
        if self.training:
            self.log_mel_min.fill_(min(self.log_mel_min, log_mel.min().item()))
            self.log_mel_max.fill_(max(self.log_mel_max, log_mel.max().item()))
        return (log_mel - self.log_mel_min) / (self.log_mel_max - self.log_mel_min)


class DDSPVocoderCLI(LightningCLI):
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

        # parser.set_defaults(
        #     {
        #         "model.mel_model": {
        #             "class_path": "models.mel.Mel2Control",
        #         },
        #         "model.glottal": {
        #             "class_path": "models.synth.GlottalFlowTable",
        #         },
        #         "model.mel_trsfm": {
        #             "class_path": "WrappedMelSpectrogram",
        #         },
        #         "model.voice_ttspn": {
        #             "class_path": "models.tspn.TTSPNEncoder",
        #             "init_args": {
        #                 "out_channels": 2,
        #             },
        #         },
        #         "model.noise_ttspn": {
        #             "class_path": "models.tspn.TTSPNEncoder",
        #             "init_args": {
        #                 "out_channels": 2,
        #             },
        #         },
        #     }
        # )


class DDSPVocoder(pl.LightningModule):
    def __init__(
        self,
        encoder: VocoderParameterEncoderInterface,
        decoder: HarmonicPlusNoiseSynth,
        feature_trsfm: ScaledLogMelSpectrogram,
        criterion: nn.Module,
        window: str = "hanning",
        sample_rate: int = 24000,
        hop_length: int = 120,
        detach_f0: bool = False,
        train_with_true_f0: bool = False,
        l1_loss_weight: float = 0.0,
        f0_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.feature_trsfm = feature_trsfm

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.l1_loss_weight = l1_loss_weight
        self.f0_loss_weight = f0_loss_weight
        self.detach_f0 = detach_f0
        self.train_with_true_f0 = train_with_true_f0

    def forward(self, feats: torch.Tensor):
        (
            f0,
            harm_osc_params,
            harm_filt_params,
            noise_filt_params,
            noise_params,
        ) = self.encoder(feats)

        phase = f0 / self.sample_rate
        ctx = TimeContext(self.hop_length)

        return f0, self.decoder(
            phase,
            harm_osc_params,
            harm_filt_params,
            noise_filt_params,
            ctx=ctx,
            noise_params=noise_params,
        )

    def f0_loss(self, f0_hat, f0):
        return F.l1_loss(torch.log(f0_hat + 1e-3), torch.log(f0 + 1e-3))

    def training_step(self, batch, batch_idx):
        x, f0_in_hz = batch
        low_res_f0 = f0_in_hz[:, :: self.hop_length]

        mask = f0_in_hz > 50
        low_res_mask = mask[:, :: self.hop_length]

        feats = self.feature_trsfm(x)
        (
            f0_hat,
            harm_osc_params,
            harm_filt_params,
            noise_filt_params,
            noise_params,
        ) = self.encoder(feats)

        minimum_length = min(f0_hat.shape[1], low_res_f0.shape[1])
        low_res_f0 = low_res_f0[:, :minimum_length]
        low_res_mask = low_res_mask[:, :minimum_length]
        f0_hat = f0_hat[:, :minimum_length]

        if self.train_with_true_f0:
            phase = torch.where(low_res_mask, low_res_f0, f0_hat) / self.sample_rate
        else:
            phase = (f0_hat.detach() if self.detach_f0 else f0_hat) / self.sample_rate

        ctx = TimeContext(self.hop_length)
        x_hat = self.decoder(
            phase,
            harm_osc_params,
            harm_filt_params,
            noise_filt_params,
            ctx=ctx,
            noise_params=noise_params,
        )

        x = x[..., : x_hat.shape[-1]]
        mask = mask[:, : x_hat.shape[1]]
        loss = self.criterion(x_hat, x)
        l1_loss = torch.sum(mask.float() * (x_hat - x).abs()) / mask.count_nonzero()

        f0_loss = self.f0_loss(f0_hat[low_res_mask], low_res_f0[low_res_mask])

        self.log("train_l1_loss", l1_loss, prog_bar=False, sync_dist=True)
        self.log("train_f0_loss", f0_loss, prog_bar=False, sync_dist=True)
        if self.l1_loss_weight > 0:
            loss = loss + l1_loss * self.l1_loss_weight
        if self.f0_loss_weight > 0:
            loss = loss + f0_loss * self.f0_loss_weight

        self.log("train_loss", loss, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, f0_in_hz = batch

        mask = f0_in_hz > 50
        num_nonzero = mask.count_nonzero()

        feats = self.feature_trsfm(x)
        f0_hat, x_hat = self(feats)

        x = x[..., : x_hat.shape[-1]]
        mask = mask[:, : x_hat.shape[1]]
        loss = self.criterion(x_hat, x)
        l1_loss = torch.sum(mask.float() * (x_hat - x).abs()) / num_nonzero

        f0_in_hz = f0_in_hz[:, :: self.hop_length]
        f0_mask = mask[:, :: self.hop_length]
        minimum_length = min(f0_hat.shape[1], f0_in_hz.shape[1])
        f0_in_hz = f0_in_hz[:, :minimum_length]
        f0_mask = f0_mask[:, :minimum_length]
        f0_hat = f0_hat[:, :minimum_length]
        f0_loss = self.f0_loss(f0_hat[f0_mask], f0_in_hz[f0_mask])

        if self.l1_loss_weight > 0:
            loss = loss + l1_loss * self.l1_loss_weight
        if self.f0_loss_weight > 0:
            loss = loss + f0_loss * self.f0_loss_weight

        return loss, l1_loss, f0_loss

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_l1_loss = sum(x[1] for x in outputs) / len(outputs)
        avg_f0_loss = sum(x[2] for x in outputs) / len(outputs)

        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_l1_loss", avg_l1_loss, prog_bar=False, sync_dist=True)
        self.log("val_f0_loss", avg_f0_loss, prog_bar=False, sync_dist=True)
