import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Tuple, Callable, Union
from torchaudio.transforms import MelSpectrogram
import numpy as np
import yaml
from importlib import import_module
from frechet_audio_distance import FrechetAudioDistance

from models.utils import get_window_fn
from models.hpn import HarmonicPlusNoiseSynth
from models.sf import SourceFilterSynth
from models.enc import VocoderParameterEncoderInterface
from models.utils import AudioTensor, get_f0, freq2cent


class ScaledLogMelSpectrogram(MelSpectrogram):
    def __init__(self, window: str, **kwargs):
        super().__init__(window_fn=get_window_fn(window), **kwargs)

        self.register_buffer("log_mel_min", torch.tensor(torch.inf))
        self.register_buffer("log_mel_max", torch.tensor(-torch.inf))

    def forward(self, waveform: Tensor) -> Tensor:
        mel = super().forward(waveform).transpose(-1, -2)
        mel = AudioTensor(mel, hop_length=self.hop_length)
        log_mel = torch.log(mel + 1e-8)
        if self.training:
            self.log_mel_min.fill_(min(self.log_mel_min, torch.min(log_mel).item()))
            self.log_mel_max.fill_(max(self.log_mel_max, torch.max(log_mel).item()))
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
        decoder: Union[SourceFilterSynth, HarmonicPlusNoiseSynth],
        feature_trsfm: ScaledLogMelSpectrogram,
        criterion: nn.Module,
        encoder_class_path: str = "models.enc.VocoderParameterEncoderInterface",
        encoder_init_args: Dict = {},
        window: str = "hanning",
        sample_rate: int = 24000,
        hop_length: int = 120,
        detach_f0: bool = False,
        detach_voicing: bool = False,
        train_with_true_f0: bool = False,
        l1_loss_weight: float = 0.0,
        f0_loss_weight: float = 1.0,
        voicing_loss_weight: float = 1.0,
        inverse_target: bool = False,
    ):
        super().__init__()

        # self.save_hyperparameters()

        self.decoder = decoder
        self.criterion = criterion
        self.feature_trsfm = feature_trsfm

        module_path, class_name = encoder_class_path.rsplit(".", 1)
        module = import_module(module_path)
        split_sizes, trsfms = self.decoder.get_split_sizes_and_trsfms()
        self.encoder = getattr(module, class_name)(
            split_sizes=split_sizes, trsfms=trsfms, **encoder_init_args
        )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.l1_loss_weight = l1_loss_weight
        self.f0_loss_weight = f0_loss_weight
        self.voicing_loss_weight = voicing_loss_weight
        self.detach_f0 = detach_f0
        self.detach_voicing = detach_voicing
        self.train_with_true_f0 = train_with_true_f0
        self.inverse_target = inverse_target

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

    def f0_loss(self, f0_hat, f0):
        return F.l1_loss(torch.log(f0_hat + 1e-3), torch.log(f0 + 1e-3))

    def on_train_start(self) -> None:
        if self.logger is not None:
            self.logger.watch(self.encoder, log_freq=1000, log="all", log_graph=False)
            if len(tuple(self.decoder.parameters())) > 0:
                self.logger.watch(
                    self.decoder, log_freq=1000, log="all", log_graph=False
                )

    def on_train_end(self) -> None:
        if self.logger is not None:
            self.logger.experiment.unwatch(self.encoder)
            if len(tuple(self.decoder.parameters())) > 0:
                self.logger.experiment.unwatch(self.decoder)

    def training_step(self, batch, batch_idx):
        x, f0_in_hz = batch
        low_res_f0 = f0_in_hz[:, :: self.hop_length]

        mask = f0_in_hz > 50
        low_res_mask = mask[:, :: self.hop_length]

        feats = self.feature_trsfm(x)
        (f0_hat, *other_params, voicing_logits) = self.encoder(feats)

        minimum_length = min(f0_hat.shape[1], low_res_f0.shape[1])
        low_res_f0 = low_res_f0[:, :minimum_length]
        low_res_mask = low_res_mask[:, :minimum_length]
        f0_hat = f0_hat[:, :minimum_length]

        if voicing_logits is not None:
            voicing_logits = voicing_logits[:, :minimum_length]
            voicing = torch.sigmoid(
                torch.detach(voicing_logits) if self.detach_voicing else voicing_logits
            )
        else:
            voicing = None

        f0_for_decoder = torch.detach(f0_hat) if self.detach_f0 else f0_hat

        if self.train_with_true_f0:
            phase = (
                torch.where(low_res_mask, low_res_f0, f0_for_decoder) / self.sample_rate
            )
        else:
            phase = f0_for_decoder / self.sample_rate

        if self.inverse_target:
            x_hat, invesre_x = self.decoder(
                phase,
                *other_params,
                voicing=voicing,
                target=AudioTensor(x),
            )
            x_hat = x_hat.as_tensor()
            x = invesre_x.as_tensor()
        else:
            x_hat = self.decoder(
                phase,
                *other_params,
                voicing=voicing,
            ).as_tensor()
        # f0_hat = f0_hat.as_tensor().rename(None)

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

        if voicing is not None:
            voicing_loss = F.binary_cross_entropy_with_logits(
                voicing_logits, low_res_mask.float()
            )
            self.log("train_voicing_loss", voicing_loss, prog_bar=False, sync_dist=True)
            if self.voicing_loss_weight > 0:
                loss = loss + voicing_loss

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.tmp_val_outputs = []

    def validation_step(self, batch, batch_idx):
        x, f0_in_hz = batch

        mask = f0_in_hz > 50
        num_nonzero = mask.count_nonzero()

        feats = self.feature_trsfm(x)
        f0_hat, x_hat, voicing = self(feats)
        f0_hat = f0_hat.as_tensor()
        x_hat = x_hat.as_tensor()

        x_hat = x_hat[:, : x.shape[-1]]
        x = x[..., : x_hat.shape[-1]]
        mask = mask[:, : x_hat.shape[1]]
        loss = self.criterion(x_hat, x)
        l1_loss = torch.sum(mask.float() * (x_hat - x).abs()) / num_nonzero

        f0_in_hz = f0_in_hz[:, :: self.hop_length]
        f0_mask = mask[:, :: self.hop_length]
        minimum_length = min(f0_hat.shape[1], f0_in_hz.shape[1], f0_mask.shape[1])
        f0_in_hz = f0_in_hz[:, :minimum_length]
        f0_mask = f0_mask[:, :minimum_length]
        f0_hat = f0_hat[:, :minimum_length]
        f0_loss = self.f0_loss(f0_hat[f0_mask], f0_in_hz[f0_mask])

        if self.l1_loss_weight > 0:
            loss = loss + l1_loss * self.l1_loss_weight
        if self.f0_loss_weight > 0:
            loss = loss + f0_loss * self.f0_loss_weight

        if voicing is not None:
            voicing = voicing.as_tensor()[:, :minimum_length]
            voicing_loss = F.binary_cross_entropy(voicing, f0_mask.float())
            self.log("val_voicing_loss", voicing_loss, prog_bar=False, sync_dist=True)
            if self.voicing_loss_weight > 0:
                loss = loss + voicing_loss

            self.tmp_val_outputs.append((loss, l1_loss, f0_loss, voicing_loss))

            return loss

        self.tmp_val_outputs.append((loss, l1_loss, f0_loss))

        return loss

    def on_validation_epoch_end(self) -> None:
        outputs = self.tmp_val_outputs
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_l1_loss = sum(x[1] for x in outputs) / len(outputs)
        avg_f0_loss = sum(x[2] for x in outputs) / len(outputs)

        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_l1_loss", avg_l1_loss, prog_bar=False, sync_dist=True)
        self.log("val_f0_loss", avg_f0_loss, prog_bar=False, sync_dist=True)

        if len(outputs[0]) > 3:
            avg_voicing_loss = sum(x[3] for x in outputs) / len(outputs)
            self.log(
                "val_voicing_loss", avg_voicing_loss, prog_bar=False, sync_dist=True
            )
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

        feats = self.feature_trsfm(x)
        _, x_hat, _ = self(feats)
        x_hat = x_hat.as_tensor()

        x = x[..., : x_hat.shape[-1]]
        mss_loss = self.criterion(x_hat, x).item()

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

        self.tmp_test_outputs.append((mss_loss, f0_loss, x_true_embs, x_hat_embs, N))

        return mss_loss, f0_loss, x_true_embs, x_hat_embs, N

    def on_test_epoch_end(self) -> None:
        outputs = self.tmp_test_outputs
        weights = [x[4] for x in outputs]
        avg_mss_loss = np.average([x[0] for x in outputs], weights=weights)
        avg_f0_loss = np.average([x[1] for x in outputs], weights=weights)

        x_true_embs = np.concatenate([x[2] for x in outputs], axis=0)
        x_hat_embs = np.concatenate([x[3] for x in outputs], axis=0)

        mu_background, sigma_background = self.frechet.calculate_embd_statistics(
            x_hat_embs
        )
        mu_eval, sigma_eval = self.frechet.calculate_embd_statistics(x_true_embs)
        fad_score = self.frechet.calculate_frechet_distance(
            mu_background, sigma_background, mu_eval, sigma_eval
        )

        self.log_dict(
            {
                "avg_mss_loss": avg_mss_loss,
                "avg_f0_loss": avg_f0_loss,
                "fad_score": fad_score,
            },
            prog_bar=True,
            sync_dist=True,
        )
        delattr(self, "tmp_test_outputs")
        return
