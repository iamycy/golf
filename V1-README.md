# The GOLF vocoder for singing voice synthesis

> **_Note:_** The latest version of the code should be capable of loading the old checkpoints (under `ckpts/ismir23/ismir23`), but the training for the v1 vocoder is not guaranteed to work. We're working on it. If you want to use the old code base that was made for the ISMIR 2023 paper, please checkout the [ismir23](https://github.com/yoyololicon/golf/releases/tag/ismir23) tag or commit `6d323da`.


## Data preparation

### MPop600

1. Download the [MPop600](https://ieeexplore.ieee.org/document/9306461) dataset. The dataset is conducted in a _download-by-request_ manner. Please contact their third author [Yi-Jhe Lee](mailto:neil@master-tones.com) to get the raw files.
3. Run the following command to resample the dataset to 24 kHz wave files. 
```bash
python scripts/resample_dir.py **/f1/ output_dir --sr 24000
```
4. Extract the foundamental frequency (F0). The f0s will be saved as `.pv` file in the same directory with the original files using 5 ms hop size.
```bash
python scripts/wav2f0.py output_dir
```

## Training
(The following command has not been tested yet. See the note above.)
```bash
python main.py fit --config config.yaml --dataset.wav_dir output_dir
```

## Evaluation

### Objective Evaluation

```bash
python main.py test --config config.yaml --ckpt_path checkpoint.ckpt --data.duration 6 --data.overlap 0 --data.batch_size 16 --trainer.logger false
```

Due to incremental changes and improvements I made since 2023, the evaluation results **may not be exactly the same** as the ones reported in the ISMIR paper.
Below are the results I got from the latest version of the code using the checkpoints from the ISMIR paper:

#### f1

| Model | MSSTFT | MAE-f0 (cent) | FAD |
| ----- |:------:|:-------------:|:---:|
| DDSP | 3.09 | ~~74.47~~ 65.36 | 
| SawSing | 3.12 | ~~78.91~~ 82.25 |
| GOLF | ~~3.21~~ 3.35 | ~~77.06~~ 64.22 | 
| PULF | ~~3.27~~ 3.29 | ~~76.90~~ 67.66 |

#### m1

| Model | MSSTFT | MAE-f0 (cent) | FAD |
| ----- |:------:|:-------------:|:---:|
| DDSP | 3.12 | ~~52.95~~ 47.16 | 
| SawSing | 3.13 | ~~56.46~~ 58.66 | 
| GOLF | ~~3.26~~ 3.37 | ~~54.09~~ 49.11 |
| PULF | ~~3.35~~ 3.41 | ~~54.60~~  49.92 |

### Real-Time Factor

```bash
python test_rtf.py config.yaml checkpoint.ckpt test.wav
```

## Checkpoints

### Female(f1)

> **_Note:_** Please use the checkpoints with `*converted*` in the name for the latest version of the code.

- [DDSP](ckpts/ismir23/ddsp_f1/)
- [SawSing](ckpts/ismir23/sawsing_f1/)
- [GOLF](ckpts/ismir23/glottal_d_f1/)
- [PULF](ckpts/ismir23/pulse_f1/)

### Male(m1)

- [DDSP](ckpts/ismir23/ddsp_m1/)
- [SawSing](ckpts/ismir23/sawsing_m1/)
- [GOLF](ckpts/ismir23/glottal_d_m1/)
- [PULF](ckpts/ismir23/pulse_m1/)


## Additional links

- [Compute MUSHURA scores given the rating file from GO Listen](notebooks/ismir/mushura.ipynb)
- [Time-domain l2 experiment](notebooks/ismir/time_l2.ipynb)


## Citation

If you find this code useful, please consider citing the following paper:

```bibtex
@inproceedings{ycy2023golf,
    title = {Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables},
    author = {Yu, Chin-Yun and Fazekas, György},
    booktitle={Proc. International Society for Music Information Retrieval},
    year={2023},
    pages={667--675}
}
```