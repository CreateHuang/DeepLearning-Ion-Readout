# DL For 300-Qubit Readout

PyTorch implementation for multi-ion readout on sCMOS images.

This repository provides:

- supervised segmentation and ion-counting models
- curriculum learning with noise ramping
- training and evaluation scripts
- a lightweight inference demo

![Example output](MobileUNET_example.png)


## Model weights

Trained model weights are available in [`model_weights/`](model_weights/), including DWNetV2, StandardUNet, ViTSeg, psfanet, and seed123 result folders.

- Details and checksum: [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md)
- Release mirror for seed123: [seed123_model_weights.zip](https://github.com/CreateHuang/DeepLearning-Ion-Readout/releases/download/seed123-weights/seed123_model_weights.zip)

## Features

- Binary segmentation of ion readout images
- Centroid-aware loss for count-sensitive training
- Curriculum sampler for easy-to-hard training
- Support for multiple backbones, including DWNetV2 and SegFormer-B5
- Evaluation utilities for connected-component based ion counting

## Repository structure

```text
DL_For_300Qubit_Readout/
├── config.py
├── dataset.py
├── loss.py
├── train_main.py
├── eval_demo.py
├── Dataset_eval.py
├── trainer.py
├── curriculum.py
├── nets/
└── Pre_train/
```

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- opencv-python
- matplotlib
- tensorboardX

## Dataset layout

The training code expects a paired image/mask structure:

```text
YourDataset/
├── images/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── masks/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

Images and masks must share the same file name.

## Configuration

Edit `config.py` to point to your dataset root. The default loader in `dataset.py` reads from the path configured there.

If you use the provided scripts as-is, also check the hard-coded paths in:

- `train_main.py`
- `eval_demo.py`
- `run_dwnet_seed_training.ps1`
- `Save_weights2txt.py`

## Training

The main training entry point is `train_main.py`.

Typical usage:

```bash
python train_main.py --model dwnet --seed 1 --load_mode full --out_dir outputs/train_unet_multiion
```

Common options:

- `--model`: model name used by the project scripts
- `--seed`: random seed
- `--load_mode`: pretrained weight loading mode
- `--out_dir`: output directory for checkpoints and logs

Training outputs usually include:

- `best.pth`
- `hist.csv`
- TensorBoard logs

## Evaluation

Use `eval_demo.py` for a simple inference and visualization example.

The script loads a checkpoint, runs segmentation, counts connected components, and draws inferred ion centers on the output image.

## Pretraining utilities

The `Pre_train/` directory contains denoising-related code for pretraining and synthetic corruption experiments.

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use this code in your research, please cite
