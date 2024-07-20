import torch
import argparse
import pathlib
import torchaudio
import numpy as np
import yaml
from typing import List, Callable, Any
from importlib import import_module
from itertools import starmap, accumulate
import time

# from frechet_audio_distance import FrechetAudioDistance

from ltng.vocoder import DDSPVocoder
from ltng.ae import VoiceAutoEncoder
from models.audiotensor import AudioTensor


def get_instance(config):
    module_path, class_name = config["class_path"].rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)(**config.get("init_args", {}))


def dict2object(config: dict):
    for k in config.keys():
        v = config[k]
        if isinstance(v, dict):
            config[k] = dict2object(v)
    if "class_path" in config:
        return get_instance(config)
    return config


def load_ismir_ckpt(model_configs, ckpt_path, device):
    model_configs["feature_trsfm"]["init_args"]["sample_rate"] = model_configs[
        "sample_rate"
    ]
    model_configs["feature_trsfm"]["init_args"]["window"] = model_configs["window"]
    model_configs["feature_trsfm"]["init_args"]["hop_length"] = model_configs[
        "hop_length"
    ]

    def contains(d: dict, s: str) -> bool:
        for k in d.keys():
            if s in k:
                return True
            if isinstance(d[k], dict):
                if contains(d[k], s):
                    return True
            elif isinstance(d[k], str):
                if s in d[k]:
                    return True
        return False

    if contains(model_configs, "DownsampledIndexedGlottalFlowTable"):
        # GOLF
        swap_weights = lambda voice_lpc, voice_gain, noise_lpc, noise_gain, h: (
            h,
            voice_gain,
            voice_lpc,
            noise_gain,
            noise_lpc,
        )

        lpc_order = model_configs["decoder"]["init_args"]["harm_filter"]["init_args"][
            "lpc_order"
        ]
        h_size = model_configs["decoder"]["init_args"]["harm_oscillator"]["init_args"][
            "in_channels"
        ]

        old_split_sizes = [lpc_order, 1, lpc_order, 1, h_size]
    elif contains(model_configs, "AdditivePulseTrain") and contains(
        model_configs, "LTVMinimumPhaseFilter"
    ):
        # PULF
        swap_weights = lambda voice_lpc, voice_gain, noise_lpc, noise_gain: (
            voice_gain,
            voice_lpc,
            noise_gain,
            noise_lpc,
        )
        harm_lpc_order = model_configs["decoder"]["init_args"]["harm_filter"][
            "init_args"
        ]["lpc_order"]
        noise_lpc_order = model_configs["decoder"]["init_args"]["noise_filter"][
            "init_args"
        ]["lpc_order"]
        old_split_sizes = [harm_lpc_order, 1, noise_lpc_order, 1]
    else:
        swap_weights = old_split_sizes = None

    model_configs = dict2object(model_configs)
    model = DDSPVocoder(**model_configs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # remove "_kernel" from key
    state_dict = {
        k: v for k, v in ckpt["state_dict"].items() if not k.endswith("_kernel")
    }
    state_dict = {
        k.replace("amplicudes", "amplitudes"): v for k, v in state_dict.items()
    }

    if old_split_sizes is not None:
        size_sum = sum(old_split_sizes)
        state_dict = {
            k: (
                torch.cat(
                    [
                        v[:-size_sum],
                        torch.cat(
                            list(
                                swap_weights(
                                    *torch.split(v[-size_sum:], old_split_sizes, dim=0)
                                )
                            ),
                            dim=0,
                        ),
                    ],
                    dim=0,
                )
                if "out_linear" in k
                else v
            )
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    ckpt["state_dict"] = state_dict
    return model, ckpt


def ismir_rtf(model_configs, ckpt_path, device, x, test_duration, num):
    # model = load_ismir_ckpt(model_configs, ckpt_path, device)
    model_configs["feature_trsfm"]["init_args"]["sample_rate"] = model_configs[
        "sample_rate"
    ]
    model_configs["feature_trsfm"]["init_args"]["window"] = model_configs["window"]
    model_configs["feature_trsfm"]["init_args"]["hop_length"] = model_configs[
        "hop_length"
    ]

    model_configs = dict2object(model_configs)
    model = DDSPVocoder(**model_configs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    model.eval()

    # get mel
    mel = model.feature_trsfm(x)

    runner = lambda: model(mel)

    measurements, _ = bench(runner, num)
    avg_synthesis_time = np.mean(measurements)
    print(f"Average synthesis time: {avg_synthesis_time:.3f}")
    print(f"Real time factor: {avg_synthesis_time / test_duration:.3f}")


def bench(runner: Callable[[], Any], num):
    results = accumulate(
        range(num),
        lambda *_: (time.time(), runner()),
        initial=(time.time(), None),
    )
    time_stamps, runner_results = zip(*results)
    time_diff = starmap(lambda x, y: y - x, zip(time_stamps, time_stamps[1:]))
    measurements = sorted(time_diff)[1:-1]
    return measurements, runner_results[-1]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        "Test model Real time factor with a given wave file"
    )
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint file")
    parser.add_argument("wav", type=str, help="Path to wav file")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of test run")
    parser.add_argument(
        "--duration", type=float, default=6.0, help="Duration of samples"
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    # load wav
    x, sr = torchaudio.load(args.wav)
    x = x[:, : int(sr * args.duration)].to(device)
    test_duration = x.shape[1] / sr
    print(f"Test duration: {test_duration:.3f}")
    x = AudioTensor(x)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_configs = config["model"]
    if "init_args" in model_configs.keys():
        model_configs = model_configs["init_args"]

    assert sr == model_configs["sample_rate"]

    if "feature_trsfm" in model_configs.keys():
        ismir_rtf(model_configs, args.ckpt, device, x, test_duration, args.num)
        return

    model_configs = dict2object(model_configs)

    model = VoiceAutoEncoder.load_from_checkpoint(
        args.ckpt, map_location=device, **model_configs
    )
    model.eval()

    f0_hop_num_frames = sr // 200
    num_f0 = x.shape[1] // f0_hop_num_frames + 1
    f0_in_hz = AudioTensor(
        torch.full((1, num_f0), 150.0, device=device), f0_hop_num_frames
    )

    def analysis():
        params = model.encoder(x, f0=f0_in_hz if model.train_with_true_f0 else None)
        f0_hat = params.pop("f0", None)
        if f0_hat is not None:
            phase = f0_hat / sr
        else:
            phase = f0_in_hz / sr
        params["phase"] = phase
        return params

    measurements, params = bench(analysis, args.num)
    avg_analysis_time = np.mean(measurements)
    print(f"Average analysis time: {avg_analysis_time:.3f}")
    print(f"Real time factor: {avg_analysis_time / test_duration:.3f}")

    def synthesis():
        y = model.decoder(**params)
        return y

    measurements, _ = bench(synthesis, args.num)
    avg_synthesis_time = np.mean(measurements)

    print(f"Average synthesis time: {avg_synthesis_time:.3f}")
    print(f"Real time factor: {avg_synthesis_time / test_duration:.3f}")

    print(f"Total time: {avg_analysis_time + avg_synthesis_time:.3f}")
    print(
        f"Total real time factor: {(avg_analysis_time + avg_synthesis_time) / test_duration:.2f}"
    )


if __name__ == "__main__":
    main()
