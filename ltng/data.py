from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from datasets.mir1k import MIR1KDataset
from datasets.mpop600 import MPop600Dataset


class MIR1K(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_dir: str,
        segment: int,
        overlap: int = 0,
        upsample_f0: bool = False,
        in_hertz: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = MIR1KDataset(
                data_dir=self.hparams.data_dir,
                segment=self.hparams.segment,
                overlap=self.hparams.overlap,
                upsample_f0=self.hparams.upsample_f0,
                in_hertz=self.hparams.in_hertz,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )


class MPop600(LightningDataModule):
    def __init__(
        self, batch_size: int, wav_dir: str, duration: float = 2, overlap: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = MPop600Dataset(
                wav_dir=self.hparams.wav_dir,
                split="train",
                duration=self.hparams.duration,
                overlap=self.hparams.overlap,
            )

        if stage == "validate" or stage == "fit":
            self.valid_dataset = MPop600Dataset(
                wav_dir=self.hparams.wav_dir,
                split="valid",
                duration=self.hparams.duration,
                overlap=self.hparams.overlap,
            )

        if stage == "test":
            self.test_dataset = MPop600Dataset(
                wav_dir=self.hparams.wav_dir,
                split="test",
                duration=self.hparams.duration,
                overlap=self.hparams.overlap,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )
