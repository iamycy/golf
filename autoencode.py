import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch import Trainer, LightningModule
from lightning.fabric.utilities.cloud_io import get_filesystem
import pathlib
import os

from ltng.ae import VoiceAutoEncoder, VoiceAutoEncoderCLI
from main import MyConfigCallback

if __name__ == "__main__":
    cli = VoiceAutoEncoderCLI(
        VoiceAutoEncoder,
        # subclass_mode_model=True,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": DDPStrategy(find_unused_parameters=False),
            "log_every_n_steps": 1,
            # "logger": WandbLogger(
            #     project="golf",
            #     log_model="all",
            # ),
        },
        save_config_callback=MyConfigCallback,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
    )
