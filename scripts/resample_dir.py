import soundfile as sf
import shutil
from multiprocessing import Pool
import soxr
import os
from tqdm import tqdm
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--suffix", type=str, default=".wav")
    args = parser.parse_args()

    src = Path(args.src)
    target = Path(args.target)

    def runner(f: Path):
        sr = sf.info(f).samplerate
        output_name = target / f.relative_to(src).with_suffix(".wav")
        output_name.parent.mkdir(parents=True, exist_ok=True)
        if sr == args.sr:
            shutil.copy2(f, output_name)
            return
        x, sr = sf.read(f)
        x = soxr.resample(x, sr, args.sr)
        sf.write(output_name, x, args.sr)

    files = list(src.rglob("*" + args.suffix))

    with Pool(processes=min(os.cpu_count(), 16)) as pool:
        list(
            tqdm(
                pool.imap_unordered(runner, files),
                total=len(files),
            )
        )
