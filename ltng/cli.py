import pathlib
import os
import torchaudio
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer, LightningModule
from lightning.fabric.utilities.cloud_io import get_filesystem
from typing import Any, Optional, Sequence, Union

from .ae import VoiceAutoEncoder
from .vocoder import DDSPVocoder


class MyPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir):
        super().__init__("batch")
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=False, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: Union[VoiceAutoEncoder, DDSPVocoder],
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

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.already_saved:
            return

        if trainer.is_global_zero:
            if trainer.logger is not None:
                trainer.logger.log_hyperparams(self.config.as_dict())
            self.already_saved = True

        self.already_saved = trainer.strategy.broadcast(self.already_saved)

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
