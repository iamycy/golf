import numpy as np
import lightning.pytorch as pl
import pyworld as pw
from typing import Dict, Tuple, Union, Optional, Sequence, Any
from collections import defaultdict
from diffsptk import MelCepstralAnalysis, STFT
import torch
import math
import torch.nn as nn
from itertools import starmap

from models.audiotensor import AudioTensor


class WORLDAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = 240,
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.criterion = criterion
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def forward(
        self,
        x: np.ndarray,
        f0: np.ndarray,
        fs: int,
        frame_period: float = 5.0,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        t = np.arange(f0.shape[0]) * frame_period / 1000
        sp = pw.cheaptrick(x, f0, t, fs)
        ap = pw.d4c(x, f0, t, fs)

        params = {"sp": sp, "ap": ap, "f0": f0}

        y = pw.synthesize(f0, sp, ap, fs, frame_period)
        return y, params

    def on_test_start(self) -> None:
        self.tmp_test_outputs = defaultdict(list)
        self.mcep = nn.Sequential(
            STFT(512, self.sample_rate // 200, 512, window="hanning"),
            MelCepstralAnalysis(34, 512, alpha=0.46),
        ).to(self.device)

        return super().on_test_start()

    def test_step(self, batch, batch_idx):
        x, f0_in_hz = batch

        f0_in_hz = f0_in_hz[:, :: self.hop_length]
        frame_period = 1000 * self.hop_length / self.sample_rate

        x_hat = torch.stack(
            list(
                starmap(
                    lambda x_, f0: torch.from_numpy(
                        self(x_, f0, self.sample_rate, frame_period)[0]
                    )
                    .to(self.device)
                    .to(x.dtype),
                    zip(x.double().cpu().numpy(), f0_in_hz.double().cpu().numpy()),
                )
            ),
            dim=0,
        )

        loss = self.criterion(
            AudioTensor(x_hat[:, : x.shape[1]]), AudioTensor(x[:, : x_hat.shape[1]])
        ).item()
        N = x_hat.shape[0]
        tmp_dict = {"loss": loss, "N": N}

        # get mceps
        x_mceps = self.mcep(x)
        x_hat_mceps = self.mcep(x_hat)
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

        x = x[0]
        f0_in_hz = f0_in_hz[0, :: self.hop_length]
        frame_period = 1000 * self.hop_length / self.sample_rate

        x_hat, enc_params = self(
            x.double().cpu().numpy(),
            f0_in_hz.double().cpu().numpy(),
            self.sample_rate,
            frame_period,
        )

        return AudioTensor(torch.from_numpy(x_hat).float().unsqueeze(0)), enc_params
