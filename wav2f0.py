import argparse
import soundfile as sf
from tqdm import tqdm
import numpy as np
import pathlib
import pyworld as pw


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get f0 from a folder of wav files")
    parser.add_argument("wav_dir", type=str, help="Path to wav files")
    parser.add_argument("--suffix", type=str, default=".pv", help="Suffix of f0 files")
    parser.add_argument("--period", type=int, default=5, help="Frame period in ms")
    parser.add_argument("--f0-floor", type=float, default=65.0, help="F0 floor in Hz")
    parser.add_argument("--f0-ceil", type=float, default=1047.0, help="F0 ceil in Hz")

    args = parser.parse_args()

    wav_dir = pathlib.Path(args.wav_dir)
    for wav_file in tqdm(list(wav_dir.glob("*.wav"))):
        x, sr = sf.read(wav_file)
        f0, _ = pw.dio(
            x,
            sr,
            f0_floor=args.f0_floor,
            f0_ceil=args.f0_ceil,
            channels_in_octave=2,
            frame_period=args.period,
        )
        f0_path = wav_file.with_suffix(args.suffix)
        np.savetxt(f0_path, f0, fmt="%f")
