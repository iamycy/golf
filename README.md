# GlOttal-flow LPC Filter (GOLF)
[![arXiv](https://img.shields.io/badge/arXiv-2306.17252-00ff00.svg)](https://arxiv.org/abs/2306.17252)

The source code of the paper [Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables](https://yoyololicon.github.io/golf-demo/), accepted at ISMIR 2023.

## Training

1. Install python requirements.

```commandline
pip install requirements.txt
```

2. Download the [MPop600](https://ieeexplore.ieee.org/document/9306461) dataset. The dataset is conducted in a _download-by-request_ manner. Please contact the authors or me to get the raw files.

3. Resample the data to 24 kHz.

```commandline
python scripts/resample_dir.py **/f1/ output_dir --sr 24000
```

4. Generate F0 labels (stored as `.pv` files).

```commandline
python scripts/wav2f0.py output_dir
```

5. Train with the configurations `config.yaml` we used in the paper (available under `ckpts/`).

```commandline
python main.py fit --config config.yaml --dataset.init_args.wav_dir output_dir
```

## Evaluation

### Objective Evaluation

```commandline
python main.py test --config config.yaml --ckpt_path checkpoint.ckpt --data.init_args.duration 6 --data.init_args.overlap 0 --data.init_args.batch_size 16
```

### Real-Time Factor

```commandline
python test_rtf.py config.yaml checkpoint.ckpt test.wav
```

### Notebooks

- [MOS](notebooks/mos.ipynb): compute MOS score given the rating file from GO Listen.
- [time-domain l2 experiment](notebooks/time_l2.ipynb): the notebook used to conduct the time-domain L2 loss ablation study in the paper.

## Pre-trained Checkpoints

### Female(f1)

- [DDSP](ckpts/ddsp_f1/)
- [SawSing](ckpts/sawsing_f1/)
- [GOLF](ckpts/glottal_d_f1/)
- [PULF](ckpts/pulse_f1/)

### Male(m1)

- [DDSP](ckpts/ddsp_m1/)
- [SawSing](ckpts/sawsing_m1/)
- [GOLF](ckpts/glottal_d_m1/)
- [PULF](ckpts/pulse_m1/)


