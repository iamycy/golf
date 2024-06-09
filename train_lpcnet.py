import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy

from ltng.lpcnet import LPCNetVocoder, LPCNetVocoderCLI
from main import MyConfigCallback


if __name__ == "__main__":
    cli = LPCNetVocoderCLI(
        LPCNetVocoder,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": DDPStrategy(find_unused_parameters=False),
            "log_every_n_steps": 1,
        },
        save_config_callback=MyConfigCallback,
    )
