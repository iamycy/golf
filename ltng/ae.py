import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Mapping, Tuple, Callable, Union, Any
import numpy as np
import yaml
import math
from collections import defaultdict
from importlib import import_module
from diffsptk import MelCepstralAnalysis, STFT

from models.audiotensor import AudioTensor
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
            **encoder_init_args,
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
        self.tmp_test_outputs = defaultdict(list)
        self.mcep = nn.Sequential(
            STFT(512, self.sample_rate // 200, 512, window="hanning"),
            MelCepstralAnalysis(34, 512, alpha=0.46),
        ).to(self.device)

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
        N = x_hat.shape[0]
        tmp_dict = {"loss": loss, "N": N}

        # get mceps
        x_mceps = self.mcep(x.as_tensor())
        x_hat_mceps = self.mcep(x_hat.as_tensor())
        x_mceps, x_hat_mceps = (
            x_mceps[:, : x_hat_mceps.shape[1]],
            x_hat_mceps[:, : x_mceps.shape[1]],
        )
        mcd = (
            10
            * math.sqrt(2)
            / math.log(10)
            * torch.linalg.norm(x_mceps - x_hat_mceps, dim=-1).mean().item()
        )
        tmp_dict["mcd"] = mcd

        list(
            map(lambda x: self.tmp_test_outputs[x].append(tmp_dict[x]), tmp_dict.keys())
        )

        return tmp_dict

    def on_test_epoch_end(self) -> None:
        outputs = self.tmp_test_outputs
        log_dict = {}
        weights = outputs["N"]
        avg_mss_loss = np.average(outputs["loss"], weights=weights)
        log_dict["avg_mss_loss"] = avg_mss_loss

        avg_mcd = np.average(outputs["mcd"], weights=weights)
        log_dict["avg_mcd"] = avg_mcd

        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=True,
        )
        delattr(self, "tmp_test_outputs")
        return

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, f0_in_hz, rel_path = batch

        assert x.shape[0] == 1, "batch size must be 1 for inference"

        # convert to AudioTensor
        x = AudioTensor(x)
        f0_in_hz = AudioTensor(f0_in_hz)

        if self.train_with_true_f0:
            phase = torch.where(f0_in_hz == 0, 150, f0_in_hz) / self.sample_rate
            x_hat, enc_params = self(x, f0_in_hz, {"phase": phase})
        else:
            x_hat, enc_params = self(x)

        return x_hat, enc_params
