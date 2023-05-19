from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers.wandb import WandbLogger

from ltng.vocoder import DDSPVocoder, DDSPVocoderCLI


if __name__ == "__main__":
    cli = DDSPVocoderCLI(
        DDSPVocoder,
        # subclass_mode_model=True,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": DDPStrategy(find_unused_parameters=False),
            "log_every_n_steps": 1,
            "logger": WandbLogger(
                project="golf",
                log_model="all",
            ),
        },
        save_config_kwargs={
            "overwrite": True,
            "config_filename": "_config.yaml"
        },
    )
