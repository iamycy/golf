from pytorch_lightning.strategies import DDPStrategy

from ltng.vocoder import MelVocoderCLI, MelGlottalVocoder


if __name__ == "__main__":
    cli = MelVocoderCLI(
        MelGlottalVocoder,
        # subclass_mode_model=True,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": DDPStrategy(find_unused_parameters=False),
            "log_every_n_steps": 1,
        },
        # save_config_kwargs={
        #     "overwrite": True,
        # },
    )
