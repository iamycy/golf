from ltng.vocoder import DDSPVocoder, DDSPVocoderCLI
from ltng.cli import MyConfigCallback

if __name__ == "__main__":
    cli = DDSPVocoderCLI(
        DDSPVocoder,
        # subclass_mode_model=True,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": {
                "class_path": "lightning.pytorch.strategies.DDPStrategy",
                "init_args": {
                    "find_unused_parameters": False,
                },
            },
            "log_every_n_steps": 1,
        },
        save_config_callback=MyConfigCallback,
        # save_config_kwargs={"overwrite": False, "config_filename": "config.yaml"},
    )
