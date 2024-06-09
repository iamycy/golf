import argparse
from multiprocessing import Pool
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from soxr import resample
import numpy as np
from pesq import pesq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_dir", type=str)
    parser.add_argument("pred_dir", type=str)
    parser.add_argument("--suffix", type=str, default="mic1.wav")

    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)

    # exclude parent path containing "convert"
    pred_files = list(
        filter(
            lambda x: "convert" not in x.relative_to(pred_dir).parts,
            pred_dir.rglob("*" + args.suffix),
        )
    )
    ref_files = list(ref_dir / f.relative_to(pred_dir) for f in pred_files)

    def read_and_resample(path):
        audio, sr = sf.read(path)
        return resample(audio, sr, 16000)

    def runner(args):
        ref, pred = args
        return pesq(16000, ref, pred)

    with Pool(processes=8) as pool:
        pred_audios, ref_audios = list(
            zip(
                *list(
                    tqdm(
                        zip(
                            pool.imap(read_and_resample, pred_files),
                            pool.imap(read_and_resample, ref_files),
                        ),
                        total=len(pred_files),
                    )
                )
            )
        )

        pesq_scores = np.array(
            list(
                tqdm(
                    pool.imap_unordered(runner, zip(ref_audios, pred_audios)),
                    total=len(ref_audios),
                )
            )
        )

    print(f"PESQ: mean {np.mean(pesq_scores):.4f}, std {np.std(pesq_scores):.4f}")
