# GlOttal-flow LPC Filter (GOLF)
[![arXiv](https://img.shields.io/badge/arXiv-2306.17252-00ff00.svg)](https://arxiv.org/abs/2306.17252)

The accompanying code for the paper [Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis]() (accepted at Interspeech 2024) and [Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables](https://zenodo.org/records/10265377) (published at ISMIR 2023).

## Training

The instructions on how to train and evaluate the model will be provided soon.

## Evaluation

## Checkpoints

## Notes

- The latest version of the code should be capable of loading the old checkpoints (under `ckpts/`), but the training for the v1 vocoder is not guaranteed to work. If you want to use the old code base that was made for the ISMIR 2023 paper, please checkout the [ismir23](https://github.com/yoyololicon/golf/releases/tag/ismir23) tag or commit `6d323da`.


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
```