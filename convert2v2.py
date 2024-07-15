import argparse
import yaml
import torch

from test_rtf import load_ismir_ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Convert a v1 checkpoint to v2 checkpoint"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the config file",
    )
    parser.add_argument(
        "v1",
        type=str,
        help="Path to the v1 checkpoint",
    )
    parser.add_argument(
        "v2",
        type=str,
        help="Path to save the v2 checkpoint",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_configs = config["model"]
    _, ckpt = load_ismir_ckpt(model_configs, args.v1, "cpu")

    torch.save(ckpt, args.v2)


if __name__ == "__main__":
    main()
