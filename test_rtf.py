import torch
import argparse
import pathlib
import torchaudio
import numpy as np
from tqdm import tqdm
import yaml
from typing import List
from importlib import import_module
import pyworld as pw
from functools import partial
import time
from frechet_audio_distance import FrechetAudioDistance

from datasets.mpop600 import MPop600Dataset
from loss.spec import MSSLoss
from ltng.vocoder import DDSPVocoder


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

    model_configs["feature_trsfm"]["init_args"]["sample_rate"] = model_configs[
        "sample_rate"
    ]
    model_configs["feature_trsfm"]["init_args"]["window"] = model_configs["window"]
    model_configs["feature_trsfm"]["init_args"]["hop_length"] = model_configs[
        "hop_length"
    ]
    model_configs = dict2object(model_configs)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = DDSPVocoder.load_from_checkpoint(args.ckpt, **model_configs).to(device)
    model.eval()

    # load wav
    x, sr = torchaudio.load(args.wav)
    assert sr == model_configs["sample_rate"]
    x = x[:, : int(sr * args.duration)].to(device)

    # get mel
    mel = model.feature_trsfm(x)

    measurements = []
    for _ in range(args.num):
        start = time.time()
        f0_hat, *_, x_hat = model(mel)
        end = time.time()
        measurements.append(end - start)

    # drop lowest and highest
    measurements.sort()
    measurements = measurements[1:-1]

    print(f"Average time: {np.mean(measurements)}")
    print(f"Real time factor: {np.mean(measurements) / args.duration}")


if __name__ == "__main__":
    main()
