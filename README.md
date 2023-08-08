# COFGAN

### Introduction
This is a PyTorch reimplementation of **COFGAN**: GAN-based Video Super-Resolution Method by optiCal flOw-Free Motion Estimation and Compensation. 

Supported model training/testing on the [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset.


### Contents
1. [Dependencies](#dependencies)
1. [Training](#training)
1. [Testing](#testing)
1. [License & Citation](#license--citation)
1. [Acknowledgements](#acknowledgements)



## Dependencies
- Ubuntu >= 16.04
- NVIDIA GPU + CUDA
- Python >= 3.7
- PyTorch >= 1.4.0
- Python packages: numpy, matplotlib, opencv-python, pyyaml, lmdb

## Training
For model training, we use REDS as the training dataset, just download it from [here](https://seungjunnah.github.io/Datasets/reds.html).The following steps are for 4x upsampling for BD degradation. You can switch to 2x upsampling.

1. Download the official training dataset and rename to `REDS/Raw`, and place under `./data`.

2. Generate LMDB for GT data to accelerate IO. The LR counterpart will then be generated on the fly during training.
```bash
python ./scripts/create_lmdb.py --dataset REDS --raw_dir ./data/REDS/Bicubic4xLR --lmdb_dir ./data/REDS/Bicubic4xLR.lmdb
```

The following shows the dataset structure after finishing the above two steps.
```tex
data
  ├─ REDS
    ├─ Raw                 # Raw dataset
      ├─ 000
        └─ ***.png
      ├─ 001
        └─ ***.png
      └─ ...
    └─ GT.lmdb             # LMDB dataset
      ├─ data.mdb
      ├─ lock.mdb
      └─ meta_info.pkl     # each key has format: [vid]_[total_frame]x[h]x[w]_[i-th_frame]
```

3. Train a COFFEE model first, which can provide a better initialization for the subsequent COFGAN training. COFFEE has the same generator as COFGAN, but without perceptual training (GAN and perceptual losses).
```bash
bash ./train.sh BD FRVSR/FRVSR_REDS_4xSR_2GPU
```

When the training is complete, set the generator's `load_path` in `experiments_BD/TecoGAN/TecoGAN_REDS_4xSR_2GPU/train.yml` to the latest checkpoint weight of the COFFEE model.

4. Train a COFGAN model. You can specify which gpu to be used in `train.sh`. By default, the training is conducted in the background and the output info will be logged in `./experiments_BD/TecoGAN/TecoGAN_REDS/train/train.log`.
```bash
bash ./train.sh BD TecoGAN/TecoGAN_REDS_4xSR_2GPU
```

5. Run the following script to monitor the training process and visualize the validation performance.
```bash
python ./scripts/monitor_training.py -dg BD -m TecoGAN/TecoGAN_REDS_4xSR_2GPU -ds Vid4
```
> Note that the validation results are NOT exactly the same as the testing results mentioned above due to different implementation of the metrics. The differences are caused by croping policy, LPIPS version and some other issues.

## Testing

**Note:** We apply different models according to the degradation type. 

1. Download the official Vid4 and ToS3 datasets.
```bash
bash ./scripts/download/download_datasets.sh BD 
```
> You can manually download these datasets from Google Drive, and unzip them under `./data`.
> * Vid4 Dataset [[Ground-Truth Data](https://drive.google.com/file/d/1T8TuyyOxEUfXzCanH5kvNH2iA8nI06Wj/view?usp=sharing)] [[Low Resolution Data (BD)](https://drive.google.com/file/d/1-5NFW6fEPUczmRqKHtBVyhn2Wge6j3ma/view?usp=sharing)]
> * ToS3 Dataset [[Ground-Truth Data](https://drive.google.com/file/d/1XoR_NVBR-LbZOA8fXh7d4oPV0M8fRi8a/view?usp=sharing)] [[Low Resolution Data (BD)](https://drive.google.com/file/d/1rDCe61kR-OykLyCo2Ornd2YgPnul2ffM/view?usp=sharing)]

The dataset structure is shown as below.
```tex
data
  ├─ Vid4
    ├─ GT                # Ground-Truth (GT) sequences
      └─ calendar
        └─ ***.png
    ├─ Gaussian4xLR      # Low Resolution (LR) sequences in BD degradation
      └─ calendar
        └─ ***.png
  └─ ToS3
    ├─ GT
    ├─ Gaussian4xLR
    └─ Bicubic4xLR
```

2. Run COFGAN for 4x SR. The results will be saved in `./results`. You can specify which model and how many gpus to be used in `test.sh`.
```bash
bash ./test.sh BD TecoGAN/TecoGAN_REDS_4xSR_2GPU
```

3. Evaluate the upsampled results using the official metrics. These codes are borrowed from [TecoGAN-TensorFlow](https://github.com/thunil/TecoGAN), with minor modifications to adapt to the BI degradation.
```bash
python ./codes/official_metrics/evaluate.py -m TecoGAN_BD_iter500000
```

4. Profile model (FLOPs, parameters and speed). You can modify the last argument to specify the size of the LR video.
```bash
bash ./profile.sh BD TecoGAN/TecoGAN_REDS_4xSR_2GPU 3x512x512
```




## License & Citation
If you use this code for your research, please cite the following paper and project.
```tex
@article{tecogan2020,
  title={Learning temporal coherence via self-supervision for GAN-based video generation},
  author={Chu, Mengyu and Xie, You and Mayer, Jonas and Leal-Taix{\'e}, Laura and Thuerey, Nils},
  journal={ACM Transactions on Graphics (TOG)},
  volume={39},
  number={4},
  pages={75--1},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```
```tex
@misc{tecogan_pytorch,
  author={Deng, Jianing and Zhuo, Cheng},
  title={PyTorch Implementation of Temporally Coherent GAN (TecoGAN) for Video Super-Resolution},
  howpublished="\url{https://github.com/skycrapers/TecoGAN-PyTorch}",
  year={2020},
}
```


## Acknowledgements
This code is built on [TecoGAN-TensorFlow](https://github.com/thunil/TecoGAN), [BasicSR](https://github.com/xinntao/BasicSR) and [LPIPS](https://github.com/richzhang/PerceptualSimilarity). We thank the authors for sharing their codes.
