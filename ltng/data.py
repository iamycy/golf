from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
import pathlib
import numpy as np
from tqdm import tqdm
import soundfile as sf
from functools import partial

from datasets.mir1k import MIR1KDataset
from datasets.mpop600 import MPop600Dataset


class LJSpeechDataset(MPop600Dataset):
    test_file_postfix = set(f"LJ001-{i:04d}.wav" for i in range(1, 21))
    valid_file_postfix = set(f"LJ001-{i:04d}.wav" for i in range(21, 101))

    def __init__(
        self,
        wav_dir: str,
        split: str = "train",
        duration: float = 2.0,
        overlap: float = 1.0,
    ):
        wav_dir = pathlib.Path(wav_dir)
        test_files = []
        valid_files = []
        train_files = []
        for f in wav_dir.glob("*.wav"):
            postfix = f.name
            if postfix in self.test_file_postfix:
                test_files.append(f)
            elif postfix in self.valid_file_postfix:
                valid_files.append(f)
            else:
                train_files.append(f)

        if split == "train":
            self.files = train_files
        elif split == "valid":
            self.files = valid_files
        elif split == "test":
            self.files = test_files
        else:
            raise ValueError(f"Unknown split: {split}")

        self.sample_rate = None

        file_lengths = []
        self.samples = []
        self.f0s = []

        print("Gathering files ...")
        for filename in tqdm(self.files):
            x, sr = sf.read(filename)
            if self.sample_rate is None:
                self.sample_rate = sr
                self.segment_num_frames = int(duration * self.sample_rate)
                self.hop_num_frames = int((duration - overlap) * self.sample_rate)
            else:
                assert sr == self.sample_rate
            f0 = np.loadtxt(filename.with_suffix(".pv"))
            # interpolate f0 to frame level
            f0 = np.interp(
                np.arange(0, len(x)),
                np.arange(0, len(f0)) * self.sample_rate * 0.005,
                f0,
            )
            f0[f0 < 80] = 0

            self.f0s.append(f0)
            self.samples.append(x)
            file_lengths.append(
                max(0, x.shape[0] - self.segment_num_frames) // self.hop_num_frames + 1
            )

        self.file_lengths = np.array(file_lengths)
        self.boundaries = np.cumsum(np.array([0] + file_lengths))


class M4SingerDataset(Dataset):
    test_folder_prefixes = set(["Alto-1", "Soprano-1", "Tenor-1", "Bass-1"])
    valid_folder_prefixes = set(["Alto-2", "Alto-3", "Tenor-2", "Tenor-3"])
    file_suffix = ".wav"

    def __init__(
        self,
        wav_dir: str,
        split: str = "train",
        duration: float = 2.0,
        overlap: float = 1.0,
        f0_suffix: str = ".pv",
    ):
        super().__init__()
        wav_dir = pathlib.Path(wav_dir)
        test_files = []
        valid_files = []
        train_files = []
        for f in wav_dir.glob("**/*" + self.file_suffix):
            parent_prefix = f.parent.name.split("#")[0]
            if parent_prefix in self.test_folder_prefixes:
                test_files.append(f)
            elif parent_prefix in self.valid_folder_prefixes:
                valid_files.append(f)
            else:
                train_files.append(f)

        if split == "train":
            self.files = train_files
        elif split == "valid":
            self.files = valid_files
        elif split == "test":
            self.files = test_files
        else:
            raise ValueError(f"Unknown split: {split}")

        self.sample_rate = None

        file_lengths = []
        self.samples = []
        self.f0s = []

        print("Gathering files ...")
        for filename in tqdm(self.files):
            x, sr = sf.read(filename, dtype="float32")
            if self.sample_rate is None:
                self.sample_rate = sr
                self.segment_num_frames = int(duration * self.sample_rate)
                self.hop_num_frames = int((duration - overlap) * self.sample_rate)
                self.f0_hop_num_frames = 0.005 * self.sample_rate
            else:
                assert sr == self.sample_rate
            f0 = np.loadtxt(filename.with_suffix(f0_suffix))

            self.f0s.append(f0)
            self.samples.append(x)
            file_lengths.append(
                max(0, x.shape[0] - self.segment_num_frames) // self.hop_num_frames + 1
            )

        self.file_lengths = np.array(file_lengths)
        self.boundaries = np.cumsum(np.array([0] + file_lengths))

    def __len__(self):
        return self.boundaries[-1]

    def __getitem__(self, index):
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        x = self.samples[bin_pos]
        f0 = self.f0s[bin_pos]
        f0 = np.where(f0 < 60, 0, f0)
        offset = (index - self.boundaries[bin_pos]) * self.hop_num_frames

        x = x[offset : offset + self.segment_num_frames]
        tp = np.arange(len(f0)) * self.f0_hop_num_frames
        t = np.arange(offset, offset + self.segment_num_frames)
        mask = np.interp(t, tp, (f0 == 0).astype(float), right=1) > 0
        interp_f0 = np.where(mask, 0, np.interp(t, tp, f0))

        if x.shape[0] < self.segment_num_frames:
            x = np.pad(x, (0, self.segment_num_frames - x.shape[0]), "constant")
        else:
            x = x[: self.segment_num_frames]
        return x.astype(np.float32), interp_f0.astype(np.float32)


class VCTKDataset(M4SingerDataset):
    test_folder_prefixes = set(
        [
            "p360",
            "p361",
            "p362",
            "p363",
            "p364",
            "p374",
            "p376",
            "s5",
        ]
    )

    valid_folder_prefixes = set(
        [
            "p225",
            "p226",
            "p227",
            "p228",
            "p229",
            "p230",
            "p231",
            "p232",
            "p233",
            "p234",
            "p236",
            "p237",
            "p238",
            "p239",
            "p240",
            "p241",
        ]
    )

    file_suffix = "mic1.wav"


class VCTKInferenceDataset(Dataset):
    def __init__(self, wav_dir: str, split: str = "train", f0_suffix: str = ".pv"):
        super().__init__()
        self.wav_dir = pathlib.Path(wav_dir)
        test_files = []
        valid_files = []
        train_files = []
        for f in self.wav_dir.glob("**/*" + VCTKDataset.file_suffix):
            parent_prefix = f.parent.name.split("#")[0]
            if parent_prefix in VCTKDataset.test_folder_prefixes:
                test_files.append(f)
            elif parent_prefix in VCTKDataset.valid_folder_prefixes:
                valid_files.append(f)
            else:
                train_files.append(f)

        if split == "train":
            self.files = train_files
        elif split == "valid":
            self.files = valid_files
        elif split == "test":
            self.files = test_files
        else:
            raise ValueError(f"Unknown split: {split}")

        self.f0_suffix = f0_suffix

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename: pathlib.Path = self.files[index]
        y, sr = sf.read(filename)
        f0 = np.loadtxt(filename.with_suffix(self.f0_suffix))
        f0 = np.where(f0 < 60, 0, f0)
        tp = np.arange(len(f0)) * sr // 200
        t = np.arange(y.shape[0])
        mask = np.interp(t, tp, (f0 == 0).astype(float), right=1) > 0
        interp_f0 = np.where(mask, 0, np.interp(t, tp, f0))

        # base file name
        rel_path = filename.relative_to(self.wav_dir)

        return y.astype(np.float32), interp_f0.astype(np.float32), str(rel_path)


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


class LJSpeech(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        wav_dir: str,
        duration: float = 2,
        overlap: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = LJSpeechDataset(
                wav_dir=self.hparams.wav_dir,
                split="train",
                duration=self.hparams.duration,
                overlap=self.hparams.overlap,
            )

        if stage == "validate" or stage == "fit":
            self.valid_dataset = LJSpeechDataset(
                wav_dir=self.hparams.wav_dir,
                split="valid",
                duration=self.hparams.duration,
                overlap=self.hparams.overlap,
            )

        if stage == "test":
            self.test_dataset = LJSpeechDataset(
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


class M4Singer(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        wav_dir: str,
        duration: float = 2,
        overlap: float = 0.5,
        f0_suffix: str = ".pv",
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        factory = partial(
            M4SingerDataset,
            wav_dir=self.hparams.wav_dir,
            duration=self.hparams.duration,
            overlap=self.hparams.overlap,
            f0_suffix=self.hparams.f0_suffix,
        )

        if stage == "fit":
            self.train_dataset = factory(split="train")

        if stage == "validate" or stage == "fit":
            self.valid_dataset = factory(split="valid")

        if stage == "test":
            self.test_dataset = factory(split="test")

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


class VCTK(M4Singer):
    def setup(self, stage=None):
        factory = partial(
            VCTKDataset,
            wav_dir=self.hparams.wav_dir,
            duration=self.hparams.duration,
            overlap=self.hparams.overlap,
            f0_suffix=self.hparams.f0_suffix,
        )

        if stage == "fit":
            self.train_dataset = factory(split="train")

        if stage == "validate" or stage == "fit":
            self.valid_dataset = factory(split="valid")

        if stage == "test":
            self.test_dataset = factory(split="test")

        if stage == "predict":
            self.predict_dataset = VCTKInferenceDataset(
                wav_dir=self.hparams.wav_dir,
                split="test",
                f0_suffix=self.hparams.f0_suffix,
            )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            drop_last=False,
        )
