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

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_configs = config["model"]
    if "init_args" in model_configs.keys():
        model_configs = model_configs["init_args"]

    # model_configs["feature_trsfm"]["init_args"]["sample_rate"] = model_configs[
    #     "sample_rate"
    # ]
    # model_configs["feature_trsfm"]["init_args"]["window"] = model_configs["window"]
    # model_configs["feature_trsfm"]["init_args"]["hop_length"] = model_configs[
    #     "hop_length"
    # ]
    model_configs = dict2object(model_configs)

    device = torch.device("cuda" if args.cuda else "cpu")

    # model = DDSPVocoder.load_from_checkpoint(args.ckpt, **model_configs).to(device)
    model = VoiceAutoEncoder.load_from_checkpoint(
        args.ckpt, map_location=device, **model_configs
    )
    model.eval()

    # load wav
    x, sr = torchaudio.load(args.wav)
    assert sr == model_configs["sample_rate"]
    x = x[:, : int(sr * args.duration)].to(device)
    test_duration = x.shape[1] / sr
    print(f"Test duration: {test_duration:.3f}")
    x = AudioTensor(x)

    f0_hop_num_frames = sr // 200
    num_f0 = x.shape[1] // f0_hop_num_frames + 1
    f0_in_hz = AudioTensor(
        torch.full((1, num_f0), 150.0, device=device), f0_hop_num_frames
    )

    # get mel
    # mel = model.feature_trsfm(x)

    def bench(runner: Callable[[], Any]):
        results = accumulate(
            range(args.num),
            lambda *_: (time.time(), runner()),
            initial=(time.time(), None),
        )
        time_stamps, runner_results = zip(*results)
        time_diff = starmap(lambda x, y: y - x, zip(time_stamps, time_stamps[1:]))
        measurements = sorted(time_diff)[1:-1]
        return measurements, runner_results[-1]

    def analysis():
        params = model.encoder(x, f0=f0_in_hz if model.train_with_true_f0 else None)
        f0_hat = params.pop("f0", None)
        if f0_hat is not None:
            phase = f0_hat / sr
        else:
            phase = f0_in_hz / sr
        params["phase"] = phase
        return params

    measurements, params = bench(analysis)
    avg_analysis_time = np.mean(measurements)
    print(f"Average analysis time: {avg_analysis_time:.3f}")
    print(f"Real time factor: {avg_analysis_time / test_duration:.3f}")

    def synthesis():
        y = model.decoder(**params)
        return y

    measurements, _ = bench(synthesis)
    avg_synthesis_time = np.mean(measurements)

    print(f"Average synthesis time: {avg_synthesis_time:.3f}")
    print(f"Real time factor: {avg_synthesis_time / test_duration:.3f}")

    print(f"Total time: {avg_analysis_time + avg_synthesis_time:.3f}")
    print(
        f"Total real time factor: {(avg_analysis_time + avg_synthesis_time) / test_duration:.2f}"
    )


if __name__ == "__main__":
    main()
