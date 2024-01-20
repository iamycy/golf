from typing import Any, Optional, Sequence
from lightning import LightningModule, Trainer
import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch import Trainer, LightningModule
import pathlib
import torchaudio
import os

from ltng.ae import VoiceAutoEncoder, VoiceAutoEncoderCLI
from main import MyConfigCallback


class MyPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir):
        super().__init__("batch")
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=False, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: VoiceAutoEncoder,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        *_, rel_path = batch
        pred, _ = prediction
        sr = pl_module.sample_rate
        out_path = self.output_dir / rel_path[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            out_path,
            pred.as_tensor().cpu(),
            sample_rate=sr,
        )


if __name__ == "__main__":
    cli = VoiceAutoEncoderCLI(
        # VoiceAutoEncoder,
        # subclass_mode_model=True,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": {
                "class_path": "lightning.pytorch.strategies.DDPStrategy",
                "init_args": {
                    "find_unused_parameters": False,
                },
            },
            "log_every_n_steps": 1,
        },
        save_config_callback=MyConfigCallback,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
    )
