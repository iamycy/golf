import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Mapping, Tuple, Callable, Union, Any
import numpy as np
import yaml
import math
from importlib import import_module
from frechet_audio_distance import FrechetAudioDistance

from models.utils import AudioTensor, get_f0, freq2cent
from models.sf import SourceFilterSynth
from models.hpn import HarmonicPlusNoiseSynth


class VoiceAutoEncoderCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        return


class VoiceAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        decoder: Union[SourceFilterSynth, HarmonicPlusNoiseSynth],
        criterion: nn.Module,
        encoder_class_path: str,
        encoder_init_args: Dict = {},
        sample_rate: int = 24000,
        detach_f0: bool = False,
        detach_voicing: bool = False,
        train_with_true_f0: bool = True,
        f0_loss_weight: float = 1.0,
        voicing_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.decoder = decoder
        self.criterion = criterion

        module_path, class_name = encoder_class_path.rsplit(".", 1)
        module = import_module(module_path)
        split_sizes, trsfms, args_keys = self.decoder.split_sizes_and_trsfms
        self.encoder = getattr(module, class_name)(
            split_sizes=split_sizes,
            trsfms=trsfms,
            args_keys=args_keys,
            **encoder_init_args
        )

        self.sample_rate = sample_rate
        self.f0_loss_weight = f0_loss_weight
        self.voicing_loss_weight = voicing_loss_weight
        self.detach_f0 = detach_f0
        self.detach_voicing = detach_voicing
        self.train_with_true_f0 = train_with_true_f0

    def forward(
        self,
        x: AudioTensor = None,
        f0: AudioTensor = None,
        params: Dict[str, Union[AudioTensor, Tuple[AudioTensor]]] = None,
    ) -> Tuple[AudioTensor, Dict[str, Union[AudioTensor, Tuple[AudioTensor]]]]:
        params = {} if params is None else params
        if x is not None:
            enc_params: Dict = self.encoder(x, f0=f0)
            params.update(enc_params)

            if "phase" not in params:
                params["phase"] = params["f0"] / self.sample_rate

            params.pop("f0", None)

            voicing_logits = params.pop("voicing_logits", None)
            if voicing_logits is not None:
                params["voicing"] = torch.sigmoid(voicing_logits)

        y = self.decoder(**params)
        return y, None if x is None else enc_params

    def f0_loss(self, f0_hat, f0):
        return F.l1_loss(torch.log(f0_hat + 1e-3), torch.log(f0 + 1e-3))

    def training_step(self, batch, batch_idx):
        x, f0_in_hz = batch

        # convert to AudioTensor
        x = AudioTensor(x)
        f0_in_hz = AudioTensor(f0_in_hz)

        params: Dict = self.encoder(x, f0=f0_in_hz if self.train_with_true_f0 else None)
        f0_hat = params.pop("f0", None)

        if self.train_with_true_f0:
            # phase = torch.where(f0_in_hz == 0, 150, f0_in_hz) / self.sample_rate
            random_f0 = (
                f0_in_hz.as_tensor().new_empty(f0_in_hz.shape[0], 1).uniform_(50, 500)
            )
            phase = torch.where(f0_in_hz == 0, random_f0, f0_in_hz) / self.sample_rate
        elif self.detach_f0:
            phase = f0_hat.new_tensor(f0_hat.as_tensor().detach()) / self.sample_rate
        else:
            phase = f0_hat / self.sample_rate
        params["phase"] = phase

        voicing_logits = params.pop("voicing_logits", None)
        if voicing_logits is not None:
            voicing = torch.sigmoid(voicing_logits)
            if self.detach_voicing:
                voicing = voicing.detach()
            params["voicing"] = voicing

        x_hat = self.decoder(**params)
        loss = self.criterion(x_hat[:, : x.shape[1]], x[:, : x_hat.shape[1]])

        if f0_hat is not None:
            f0_target = f0_in_hz[:, :: f0_hat.hop_length].as_tensor()[
                :, : f0_hat.shape[1]
            ]
            f0_pred = f0_hat.as_tensor()[:, : f0_target.shape[1]]
            mask = f0_target > 50
            f0_loss = self.f0_loss(f0_pred[mask], f0_target[mask])
            loss = loss + f0_loss * self.f0_loss_weight
            self.log("train_f0_loss", f0_loss, prog_bar=True, sync_dist=True)

        if voicing_logits is not None:
            voicing_target = (f0_in_hz > 50).float()
            voicing_target = voicing_target[
                :, :: voicing_logits.hop_length
            ].as_tensor()[:, : voicing_logits.shape[1]]
            voicing_logits = voicing_logits.as_tensor()[:, : voicing_target.shape[1]]
            voicing_loss = F.binary_cross_entropy_with_logits(
                voicing_logits, voicing_target, reduction="mean"
            )
            loss = loss + voicing_loss * self.voicing_loss_weight
            self.log("train_voicing_loss", voicing_loss, prog_bar=True, sync_dist=True)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.tmp_val_outputs = []

    def validation_step(self, batch, batch_idx):
        x, f0_in_hz = batch

        # convert to AudioTensor
        x = AudioTensor(x)
        f0_in_hz = AudioTensor(f0_in_hz)

        if self.train_with_true_f0:
            phase = torch.where(f0_in_hz == 0, 150, f0_in_hz) / self.sample_rate
            x_hat, enc_params = self(x, f0_in_hz, {"phase": phase})
        else:
            x_hat, enc_params = self(x)
        loss = self.criterion(x_hat[:, : x.shape[1]], x[:, : x_hat.shape[1]])

        val_outputs = []
        if "f0" in enc_params:
            f0_hat = enc_params["f0"]
            f0_target = f0_in_hz[:, :: f0_hat.hop_length].as_tensor()[
                :, : f0_hat.shape[1]
            ]
            f0_pred = f0_hat.as_tensor()[:, : f0_target.shape[1]]
            mask = f0_target > 50
            f0_loss = self.f0_loss(f0_pred[mask], f0_target[mask])
            loss = loss + f0_loss * self.f0_loss_weight
            val_outputs.append(f0_loss)

        if "voicing_logits" in enc_params:
            voicing_logits = enc_params["voicing_logits"]
            voicing_target = (f0_in_hz > 50).float()
            voicing_target = voicing_target[
                :, :: voicing_logits.hop_length
            ].as_tensor()[:, : voicing_logits.shape[1]]
            voicing_logits = voicing_logits.as_tensor()[:, : voicing_target.shape[1]]
            voicing_loss = F.binary_cross_entropy_with_logits(
                voicing_logits, voicing_target, reduction="mean"
            )
            loss = loss + voicing_loss * self.voicing_loss_weight
            val_outputs.append(voicing_loss)

        val_outputs.insert(0, loss)

        self.tmp_val_outputs.append(val_outputs)

        return loss

    def on_validation_epoch_end(self) -> None:
        outputs = self.tmp_val_outputs
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        if len(outputs[0]) > 1:
            avg_f0_loss = sum(x[1] for x in outputs) / len(outputs)
            self.log("val_f0_loss", avg_f0_loss, prog_bar=True, sync_dist=True)

        if len(outputs[0]) > 2:
            avg_voicing_loss = sum(x[2] for x in outputs) / len(outputs)
            self.log(
                "val_voicing_loss", avg_voicing_loss, prog_bar=True, sync_dist=True
            )

        delattr(self, "tmp_val_outputs")

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        return super().load_state_dict(state_dict, False)

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

        # convert to AudioTensor
        x = AudioTensor(x)
        f0_in_hz = AudioTensor(f0_in_hz)  # .set_hop_length(self.sample_rate // 200)

        if self.train_with_true_f0:
            phase = torch.where(f0_in_hz == 0, 150, f0_in_hz) / self.sample_rate
            x_hat, enc_params = self(x, f0_in_hz, {"phase": phase})
        else:
            x_hat, enc_params = self(x)
        loss = self.criterion(x_hat[:, : x.shape[1]], x[:, : x_hat.shape[1]]).item()

        x_hat = x_hat.as_tensor().cpu().numpy().astype(np.float64)
        x = x.as_tensor().cpu().numpy().astype(np.float64)
        N = x_hat.shape[0]

        f0_hat = np.stack(
            [get_f0(x_hat[i], self.sample_rate, f0_floor=60)[0] for i in range(N)],
            axis=0,
        )
        x_true_embs = (
            torch.cat(
                [self.frechet.model.forward(x[i], self.sample_rate) for i in range(N)],
                dim=0,
            )
            .cpu()
            .numpy()
        )
        x_hat_embs = (
            torch.cat(
                [
                    self.frechet.model.forward(x_hat[i], self.sample_rate)
                    for i in range(N)
                ],
                dim=0,
            )
            .cpu()
            .numpy()
        )

        f0_in_hz = (
            f0_in_hz.set_hop_length(self.sample_rate // 200)
            .as_tensor()
            .cpu()
            .numpy()[:, : f0_hat.shape[1]]
        )
        f0_hat = f0_hat[:, : f0_in_hz.shape[1]]
        f0_in_hz = np.maximum(f0_in_hz, 60)
        f0_hat = np.maximum(f0_hat, 60)
        f0_loss = np.mean(np.abs(freq2cent(f0_hat) - freq2cent(f0_in_hz)))

        self.tmp_test_outputs.append((loss, f0_loss, x_true_embs, x_hat_embs, N))

        return loss, f0_loss, x_true_embs, x_hat_embs, N

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
