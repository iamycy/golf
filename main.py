import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch import Trainer, LightningModule
from lightning.fabric.utilities.cloud_io import get_filesystem
import pathlib
import os

from ltng.vocoder import DDSPVocoder, DDSPVocoderCLI


class MyConfigCallback(Callback):
    def __init__(
        self,
        parser,
        config,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile
        self.already_saved = False

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.already_saved:
            return

        log_dir = pathlib.Path(trainer.checkpoint_callback.dirpath).parent
        assert log_dir is not None
        config_path = os.path.join(str(log_dir), self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )
            self.already_saved = True
            if trainer.logger is not None:
                trainer.logger.log_hyperparams(self.config.as_dict())

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)


if __name__ == "__main__":
    cli = DDSPVocoderCLI(
        DDSPVocoder,
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
        # save_config_kwargs={"overwrite": False, "config_filename": "config.yaml"},
    )
