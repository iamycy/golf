import torch
import argparse
import pathlib
import torchaudio
from tqdm import tqdm

from models.utils import get_logits2biquads
from harm_and_noise import loader

logits2biquads = get_logits2biquads("coef")


@torch.no_grad()
def get_biquads(m, x):
    h = m.feature_trsfm(x)
    enc = m.encoder

    logits = enc.backbone(h)

    coarse_split_size = list(map(sum, enc.split_sizes))

    def fn(k: str, apply_trsfm=False):
        idx = enc.args_keys.index(k)
        start = sum(coarse_split_size[:idx])
        end = start + coarse_split_size[idx]
        if end - start == 0:
            return None
        if apply_trsfm:
            return enc.trsfms[idx](
                *torch.split(logits[..., start:end], enc.split_sizes[idx], dim=-1)
            )
        return logits[..., start:end].squeeze(-1)

    def bq_fn(k: str):
        logits_slice = fn(k)
        log_gain = logits_slice[..., 0]
        biquad_logits = logits_slice[..., 1:].reshape(*logits.shape[:-1], -1, 2)
        biquads = logits2biquads(biquad_logits)
        return log_gain, biquads

    harm_log_gain, harm_biquads = bq_fn("harm_filter_params")
    noise_log_gain, noise_biquads = bq_fn("noise_filter_params")

    harm_osc_params = fn("harm_oscillator_params", apply_trsfm=True)

    voicing = logits[..., 1].sigmoid()

    if len(harm_osc_params):
        return (
            voicing,
            harm_log_gain,
            harm_biquads,
            noise_log_gain,
            noise_biquads,
            harm_osc_params[0],
        )

    return voicing, harm_log_gain, harm_biquads, noise_log_gain, noise_biquads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("ckpt", type=str)
    parser.add_argument("audio_dir", type=str)
    parser.add_argument("outfile", type=str)
    parser.add_argument("--duration", type=float, default=6.0)

    args = parser.parse_args()

    model = loader(args.config, args.ckpt)
    model.eval()

    audio_dir = pathlib.Path(args.audio_dir)

    chunk_size = int(24000 * args.duration)

    output_dict = {}
    for audio_path in tqdm(list(audio_dir.rglob("*.wav"))):
        x, sr = torchaudio.load(audio_path)
        assert sr == 24000

        for i, x_chunk in enumerate(torch.split(x, chunk_size, dim=1)):
            voicing, harm_log_gain, harm_biquads, noise_log_gain, noise_biquads, *_ = (
                get_biquads(model, x_chunk)
            )

            update_dict = {
                f"{audio_path.stem}_{i}.harm_log_gain": harm_log_gain,
                f"{audio_path.stem}_{i}.harm_biquads": harm_biquads,
                f"{audio_path.stem}_{i}.noise_log_gain": noise_log_gain,
                f"{audio_path.stem}_{i}.noise_biquads": noise_biquads,
                f"{audio_path.stem}_{i}.voicing": voicing,
            }
            if len(_):
                harm_osc_params = _[0]
                update_dict[f"{audio_path.stem}_{i}.table_select_weight"] = (
                    harm_osc_params
                )

            output_dict.update(update_dict)

    torch.save(output_dict, args.outfile)


if __name__ == "__main__":
    main()
