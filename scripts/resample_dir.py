import soundfile as sf
import shutil

# import samplerate
import soxr

# import resampy
from tqdm import tqdm
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("src", type=str)
parser.add_argument("target", type=str)
parser.add_argument("--sr", type=int, default=16000)
parser.add_argument("--suffix", type=str, default=".wav")
args = parser.parse_args()

src = Path(args.src)
target = Path(args.target)

for f in tqdm(list(src.glob("**/*" + args.suffix))):
    sr = sf.info(f).samplerate
    output_name = target / f.relative_to(src).with_suffix(".wav")
    output_name.parent.mkdir(parents=True, exist_ok=True)
    if sr == args.sr:
        shutil.copy2(f, output_name)
        continue
    x, sr = sf.read(f)
    # x = samplerate.resample(x, args.sr / sr, verbose=True)
    x = soxr.resample(x, sr, args.sr)
    sf.write(output_name, x, args.sr)
