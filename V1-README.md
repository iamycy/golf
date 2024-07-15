# The GOLF vocoder for singing voice synthesis

> **_Note:_** The latest version of the code should be capable of loading the old checkpoints (under `ckpts/ismir23`), but the training for the v1 vocoder is not guaranteed to work. We're working on it. If you want to use the old code base that was made for the ISMIR 2023 paper, please checkout the [ismir23](https://github.com/yoyololicon/golf/releases/tag/ismir23) tag or commit `6d323da`.


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