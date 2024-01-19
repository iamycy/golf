import argparse
from multiprocessing import Pool
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from soxr import resample

from pesq import pesq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_dir", type=str)
    parser.add_argument("pred_dir", type=str)
    parser.add_argument("--suffix", type=str, default="mic1.wav")

    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)

    pred_files = list(pred_dir.rglob("*" + args.suffix))
    ref_files = list(ref_dir / f.relative_to(pred_dir) for f in pred_files)

    def read_and_resample(path):
        audio, sr = sf.read(path)
        return resample(audio, sr, 16000)

    pred_audios, ref_audios = list(
        zip(
            *list(
                tqdm(
                    zip(
                        map(read_and_resample, pred_files),
                        map(read_and_resample, ref_files),
                    ),
                    total=len(pred_files),
                )
            )
        )
    )

    def runner(args):
        ref, pred = args
        return pesq(16000, ref, pred)

    with Pool(processes=8) as pool:
        pesq_score = sum(
            tqdm(
                pool.imap_unordered(runner, zip(ref_audios, pred_audios)),
                total=len(ref_audios),
            )
        ) / len(ref_audios)

    print(f"PESQ: {pesq_score}")
