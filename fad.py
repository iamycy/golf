import numpy as np
import argparse
import torch
import math
from pathlib import Path
from itertools import chain
from fadtk.model_loader import ModelLoader
from fadtk import VGGishModel
from fadtk.fad import FrechetAudioDistance, log
from fadtk.fad_batch import cache_embedding_files


class DAC24kModel(ModelLoader):
    """
    DAC model from https://github.com/descriptinc/descript-audio-codec

    pip install descript-audio-codec
    """

    def __init__(self):
        super().__init__("dac-24kHz", 1024, 24000)

    def load_model(self):
        from dac.utils import load_model

        self.model = load_model(tag="latest", model_type="24khz")
        self.model.eval()
        self.model.to(self.device)

    def _get_embedding(self, audio) -> np.ndarray:
        from audiotools import AudioSignal
        import time

        audio: AudioSignal

        # Set variables
        win_len = 5.0
        overlap_hop_ratio = 0.5

        # Fix overlap window so that it's divisible by 4 in # of samples
        win_len = ((win_len * self.sr) // 4) * 4
        win_len = win_len / self.sr
        hop_len = win_len * overlap_hop_ratio

        # Sanitize input
        audio.normalize(-16)
        audio.ensure_max_of_audio()

        nb, nac, nt = audio.audio_data.shape
        audio.audio_data = audio.audio_data.reshape(nb * nac, 1, nt)

        pad_length = math.ceil(audio.signal_duration / win_len) * win_len
        audio.zero_pad_to(int(pad_length * self.sr))
        audio = audio.collect_windows(win_len, hop_len)

        emb = []
        for i in range(audio.batch_size):
            signal_from_batch = AudioSignal(audio.audio_data[i, ...], self.sr)
            signal_from_batch.to(self.device)
            e1 = self.model.encoder(
                signal_from_batch.audio_data
            ).cpu()  # [1, 1024, timeframes]
            e1 = e1[0]  # [1024, timeframes]
            e1 = e1.transpose(0, 1)  # [timeframes, 1024]
            emb.append(e1)

        emb = torch.cat(emb, dim=0)

        return emb

    def load_wav(self, wav_file: Path):
        from audiotools import AudioSignal

        return AudioSignal(wav_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("baseline", type=str, help="The baseline dataset")
    parser.add_argument("eval", type=str, help="The directory to evaluate against")
    parser.add_argument("--csv", type=str, help="The CSV file to write results to")
    parser.add_argument("--model", type=str, choices=["dac", "vggish"], default="dac")

    # Add optional arguments
    parser.add_argument("-w", "--workers", type=int, default=4)

    args = parser.parse_args()

    match args.model:
        case "vggish":
            model = VGGishModel()
        case "dac":
            model = DAC24kModel()
        case _:
            raise ValueError("Invalid model")

    baseline = Path(args.baseline)
    eval_ = Path(args.eval)

    speaker_dirs = [x.stem for x in filter(lambda x: x.is_dir(), eval_.iterdir())]

    # calculate embedding files for each speaker
    cache_embedding_files(
        list(
            chain.from_iterable(
                [(p / s).glob("*.*") for s in speaker_dirs for p in (baseline, eval_)]
            )
        ),
        ml=model,
        workers=args.workers,
    )

    # 2. Calculate FAD of each speaker, and summarize as min, max, mean, std
    fad = FrechetAudioDistance(model, audio_load_worker=args.workers, load_model=False)

    scores = np.array(
        list(
            map(
                lambda s: fad.score(baseline / s, eval_ / s),
                speaker_dirs,
            )
        )
    )

    # 3. Print results
    log.info("FAD computed.")

    log.info(
        f"The FAD {model.name} score between {baseline} and {eval_} is: mean {scores.mean():.4f}, std {scores.std():.4f}, min {scores.min():.4f}, max {scores.max():.4f}"
    )

    # save raw scores to csv
    if args.csv:
        import pandas as pd

        pd.DataFrame.from_dict(
            dict(zip(speaker_dirs, scores)), orient="index", columns=["score"]
        ).to_csv(args.csv)
