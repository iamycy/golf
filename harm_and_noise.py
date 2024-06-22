import torch
import argparse
import pathlib
import torchaudio
import yaml
from tqdm import tqdm

from test_rtf import load_ismir_ckpt


def loader(congif_path, ckpt_path):
    with open(congif_path) as f:
        model_configs = yaml.safe_load(f)["model"]
        # print(model_configs)

    return load_ismir_ckpt(model_configs, ckpt_path, "cpu")


@torch.no_grad()
def harms_and_noise(m, x, sr):
    h = m.feature_trsfm(x)
    dec = m.decoder
    enc = m.encoder

    params = enc(h)

    f0 = params.pop("f0")
    phase = f0 / sr

    voicing_logits = params.pop("voicing_logits")
    voicing = torch.sigmoid(voicing_logits)

    harm_osc = dec.harm_oscillator(phase, *params["harm_oscillator_params"])
    harm_osc = harm_osc * voicing

    noise = dec.noise_generator(harm_osc, *params["noise_generator_params"])

    harm_osc = dec.harm_filter(harm_osc, *params["harm_filter_params"])
    noise = dec.noise_filter(noise, *params["noise_filter_params"])
    return harm_osc, noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("ckpt", type=str)
    parser.add_argument("audio_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fade", type=float, default=1.0)

    args = parser.parse_args()

    model = loader(args.config, args.ckpt)
    model.eval()

    audio_dir = pathlib.Path(args.audio_dir)
    output_dir = pathlib.Path(args.output_dir)

    chunk_size = int(24000 * args.duration)
    fade_size = int(24000 * args.fade)
    hop_size = chunk_size - fade_size
    fader = torch.linspace(0, 1, fade_size)

    for audio_path in tqdm(list(audio_dir.rglob("*.wav"))):
        x, sr = torchaudio.load(audio_path)
        assert sr == 24000

        harms = torch.zeros_like(x)
        noise = torch.zeros_like(x)
        for offset in range(0, x.shape[1], hop_size):
            x_chunk = x[:, offset : offset + chunk_size]

            harms_chunk, noise_chunk = harms_and_noise(model, x_chunk, sr)

            if offset > 0:
                actual_size = min(
                    fade_size,
                    x.shape[1] - offset,
                    harms_chunk.shape[1],
                    noise_chunk.shape[1],
                )
                harms[:, offset : offset + actual_size] *= 1 - fader[:actual_size]
                noise[:, offset : offset + actual_size] *= 1 - fader[:actual_size]

                harms_chunk[:, :actual_size] *= fader[:actual_size]
                noise_chunk[:, :actual_size] *= fader[:actual_size]

            actual_size = min(
                noise_chunk.shape[1], x.shape[1] - offset, harms_chunk.shape[1]
            )
            harms[:, offset : offset + actual_size] += harms_chunk[:, :actual_size]
            noise[:, offset : offset + actual_size] += noise_chunk[:, :actual_size]

        out_folder = output_dir / audio_path.stem
        out_folder.mkdir(exist_ok=True, parents=True)
        torchaudio.save(out_folder / "harms.wav", harms, sr)
        torchaudio.save(out_folder / "noise.wav", noise, sr)


if __name__ == "__main__":
    main()
