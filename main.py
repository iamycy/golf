from pytorch_lightning.strategies import DDPStrategy

from lightning.vocoder import MelVocoderCLI, MelGlottalVocoder


if __name__ == "__main__":
    cli = MelVocoderCLI(
        MelGlottalVocoder,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": DDPStrategy(find_unused_parameters=False),
            "log_every_n_steps": 1,
        },
    )
