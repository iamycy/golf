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

from datasets.mpop600 import MPop600Dataset
from loss.spec import MSSLoss
from ltng.vocoder import DDSPVocoder
from ltng.ae import VoiceAutoEncoder
from models.audiotensor import AudioTensor
from models.utils import ismir2interspeech_ckpt


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


def ismir_rtf(model_configs, ckpt_path, device, x, test_duration, num):
    model_configs["feature_trsfm"]["init_args"]["sample_rate"] = model_configs[
        "sample_rate"
    ]
    model_configs["feature_trsfm"]["init_args"]["window"] = model_configs["window"]
    model_configs["feature_trsfm"]["init_args"]["hop_length"] = model_configs[
        "hop_length"
    ]
    lpc_order = model_configs["decoder"]["init_args"]["harm_filter"]["init_args"][
        "lpc_order"
    ]
    h_size = model_configs["decoder"]["init_args"]["harm_oscillator"]["init_args"][
        "in_channels"
    ]

    model_configs = dict2object(model_configs)
    model = DDSPVocoder(**model_configs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # remove "_kernel" from key
    remover = lambda x: (
        {k: remover(v) for k, v in x.items() if not k.endswith("_kernel")}
        if isinstance(x, dict)
        else x
    )
    state_dict = remover(ckpt["state_dict"])
    state_dict = ismir2interspeech_ckpt(state_dict, lpc_order, h_size)
    model.load_state_dict(state_dict)
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
