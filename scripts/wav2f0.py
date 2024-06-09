import argparse
import soundfile as sf
from tqdm import tqdm
import numpy as np
import pathlib
import pyworld as pw
from penn import from_file
from pysptk import swipe as swipe_sptk
from functools import partial, reduce
from itertools import starmap, tee
import torch


def chain_functions(*functions):
    return lambda *initial: reduce(lambda x, f: f(*x), functions, initial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get f0 from a folder of wav files")
    parser.add_argument("wav_dir", type=str, help="Path to wav files")
    parser.add_argument("--method", choices=["dio", "penn", "swipe"], default="dio")
    parser.add_argument("--suffix", type=str, default=".pv", help="Suffix of f0 files")
    parser.add_argument(
        "--wav-suffix", type=str, default=".wav", help="Suffix of wav files"
    )
    parser.add_argument("--period", type=int, default=5, help="Frame period in ms")
    parser.add_argument("--f0-floor", type=float, default=65.0, help="F0 floor in Hz")
    parser.add_argument("--f0-ceil", type=float, default=1047.0, help="F0 ceil in Hz")

    args = parser.parse_args()

    if args.method == "penn":
        gpu_index = torch.cuda.current_device() if torch.cuda.is_available() else None
        inferencer = chain_functions(
            partial(
                from_file,
                hopsize=args.period / 1000,
                fmin=args.f0_floor,
                fmax=args.f0_ceil,
                center="zero",
                gpu=gpu_index,
            ),
            lambda pitch, periodicity: torch.where(periodicity > 0.065, pitch, 0.0)
            .cpu()
            .numpy()[0],
        )
    elif args.method == "dio":
        inferencer = lambda f: pw.dio(
            *sf.read(f),
            f0_floor=args.f0_floor,
            f0_ceil=args.f0_ceil,
            channels_in_octave=2,
            frame_period=args.period,
        )[0]
    elif args.method == "swipe":
        inferencer = chain_functions(
            sf.read,
            lambda x, sr: swipe_sptk(
                x,
                sr,
                hopsize=int(sr * args.period / 1000),
                min=args.f0_floor,
                max=args.f0_ceil,
                otype="f0",
            ),
        )
    else:
        raise ValueError(f"Unknown method {args.method}")

    files = pathlib.Path(args.wav_dir).rglob("*" + args.wav_suffix)
    it1, it2 = tee(files)

    list(
        tqdm(
            starmap(
                partial(np.savetxt, fmt="%f"),
                zip(
                    map(lambda f: f.with_suffix(args.suffix), it1),
                    map(inferencer, it2),
                ),
            )
        )
    )
