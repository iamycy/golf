# GlOttal-flow LPC Filter (GOLF) for Voice Synthesis
[![arXiv](https://img.shields.io/badge/arXiv-2306.17252-00ff00.svg)](https://arxiv.org/abs/2306.17252)
[![arXiv](https://img.shields.io/badge/arXiv-2406.05128-00ff00.svg)](https://arxiv.org/abs/2406.05128)

The accompanying code for the paper [Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis](https://arxiv.org/abs/2406.05128) (accepted at Interspeech 2024) and [Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables](https://zenodo.org/records/10265377) (published at ISMIR 2023).

## Data preparation

### VCTK

1. Download the VCTK 0.92 dataset from [here](https://datashare.is.ed.ac.uk/handle/10283/3443).
2. Extract the dataset to a directory, e.g., `data/vctk`.
3. Run the following command to resample the dataset to 24 kHz wave files. The resampled files will be saved in the same directory with the original files.
```bash
python scripts/resample_dir.py data/vctk --suffix .flac --sr 24000
```
4. Extract the foundamental frequency (F0). The f0s will be saved as `.pv` file in the same directory with the original files using 5 ms hop size.
```bash
python scripts/wav2f0.py data/vctk --f0-floor 60
```

## Training

Below is the command to train each models in the Interspeech [paper](https://arxiv.org/abs/2406.05128).

```bash
python autoencode.py fit --model ltng.ae.VoiceAutoEncoder --config cfg/vctk.yaml --model cfg/ae/decoder/{MODEL}.yaml --trainer.logger false
```

The `{MODEL}` corresponds to the following models:
- `ddsp` $\rightarrow$ DDSP
- `nhv` $\rightarrow$ NHV (neural homomorphic vocoder)
- `world` $\rightarrow$ $\nabla$ WORLD
- `mlsa` $\rightarrow$ MLSA (differentiable Mel-cepstral synthesis filter)
- `golf-v1` $\rightarrow$ GOLF-v1
- `golf` $\rightarrow$ GOLF-ff
- `golf-precise` $\rightarrow$ GOLF-ss

By default, the checkpoints are automatically saved under `checkpoints/` directory. 
Feel free to remove `--trainer.logger false` and edit the logger settings in the configuration file `cfg/vctk.yaml` to fit your needs.
Please checkout the LightningCLI instructions [here](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html).

## Evaluation

### MCD/MSS

After training the models, you can evaluate the models using the following command. Replace `{YOUR_CONFIG}` and `{YOUR_CHECKPOINT}` with the corresponding configuration file and checkpoint.

```bash
python autoencode.py test --model ltng.ae.VoiceAutoEncoder -c {YOUR_CONFIG}.yaml --ckpt_path {YOUR_CHECKPOINT}.ckpt --data.duration 2 --data.overlap 0 --seed_everything false --data.wav_dir data/vctk --data.batch_size 32 --trainer.logger false
```

### PESQ/FAD

For PESQ/FAD evaluation, you'll first need to store the synthesised waveforms in a directory. Replace `{YOUR_CONFIG}`, `{YOUR_CHECKPOINT}`, and `{YOUR_OUTPUT_DIR}` with the corresponding configuration file, checkpoint, and output directory.

```bash
python autoencode.py predict -c {YOUR_CONFIG}.yaml --ckpt_path {YOUR_CHECKPOINT}.ckpt --trainer.logger false --seed_everything false --data.wav_dir data/vctk --trainer.callbacks+=autoencode.MyPredictionWriter --trainer.callbacks.output_dir {YOUR_OUTPUT_DIR}
```

Make a new directory and copy the following eight speakers, which form the test set, from `data/vctk`.
```
data/vctk_test
├── p360
├── p361
├── p362
├── p363
├── p364
├── p374
├── p376
├── s5
```

Then, caculate the PESQ scores:
    
```bash
python eval_pesq.py data/vctk_test {YOUR_OUTPUT_DIR}
```

For the FAD scores:

```bash
python fad.py data/vctk_test {YOUR_OUTPUT_DIR}
```

We use [fadtk](https://github.com/microsoft/fadtk) and [descript audio codec](https://github.com/descriptinc/descript-audio-codec) for the FAD evaluation. 

### Notes on GOLF-fs

Please use the checkpoints trained with `golf.yaml` for the GOLF-fs model. Append `--model.decoder.end_filter models.filters.LTVMinimumPhaseFilterPrecise` to the evaluation commands above (`test/predict`) to use the sample-wise filter.

### Notes on non-differentiable WORLD (`pyworld`)

Please use the following commands to evaluate the non-differentiable WORLD model.

```bash
python autoencode.py test -c cfg/pyworld.yaml --data.wav_dir data/vctk --model ltng.world_ae.WORLDAutoEncoder
python autoencode.py predict -c cfg/pyworld.yaml --trainer.logger false --seed_everything false --data.wav_dir data/vctk --trainer.callbacks+=autoencode.MyPredictionWriter --trainer.callbacks.output_dir {YOUR_OUTPUT_DIR}
```

## Checkpoints

The checkpoints we used for evaluation are provided [here](ckpts/interspeech24).

## Additional links

- [Individual FAD and PESQ scores on each test speaker](https://docs.google.com/spreadsheets/d/1E_2AVUXLITRd1R5oolYvcYwKqAB5YJ_V_jWVkQKx-VQ/edit?usp=sharing)
- [MCD and MSS comparison table on W&B](https://api.wandb.ai/links/iamycy/qa1pckb0)

## Notes

- The latest version of the code should be capable of loading the old checkpoints (under `ckpts/ismir23`), but the training for the v1 vocoder is not guaranteed to work. If you want to use the old code base that was made for the ISMIR 2023 paper, please checkout the [ismir23](https://github.com/yoyololicon/golf/releases/tag/ismir23) tag or commit `6d323da`.


## Citation

If you find this code useful, please consider citing the following papers:

```bibtex
@inproceedings{ycy2023golf,
    title = {Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables},
    author = {Yu, Chin-Yun and Fazekas, György},
    booktitle={Proc. International Society for Music Information Retrieval},
    year={2023},
    pages={667--675}
}

@misc{ycy2024golf,
    title = {Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis},
    author = {Yu, Chin-Yun and Fazekas, György},
    year={2024},
    eprint={2406.05128},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```