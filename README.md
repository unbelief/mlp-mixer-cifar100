# MLP-Mixer on CIFAR-100

A course-project implementation of MLP-Mixer for CIFAR-100 image classification with CPU/small-GPU friendly settings.

## Hardware-aware design

The code automatically selects CUDA when available and otherwise runs on CPU.

Default settings are intentionally lightweight:
- CIFAR-100
- 32x32 RGB images
- patch size 4
- embedding dimension 128
- 4 Mixer blocks
- batch size 64
- 30 epochs

For very limited hardware:
python train.py --model tiny --epochs 10 --batch-size 32

For CPU-only:
python train.py --model tiny --device cpu --epochs 10 --batch-size 32

For a small GPU:
python train.py --model small --epochs 30 --batch-size 64

## Architecture

Image -> Patch Embedding -> Mixer Blocks -> Global Average Pooling -> Classifier

Each Mixer block contains:
1. Token-Mixing MLP: mixes information across image patches.
2. Channel-Mixing MLP: mixes feature channels for each patch.

The model does not use convolution inside the Mixer blocks and does not use self-attention.

## Project structure

mlp-mixer-cifar100/
  models/mlp_mixer.py
  data.py
  config.py
  utils.py
  train.py
  evaluate.py
  predict.py
  smoke_test.py
  experiments/ablation.py
  report/实验报告.md
  requirements.txt

## Installation

pip install -r requirements.txt

CIFAR-100 is downloaded automatically by torchvision on the first run.

## Training

Tiny:
python train.py --model tiny --epochs 10 --batch-size 32

Small:
python train.py --model small --epochs 30 --batch-size 64

Explicit CPU:
python train.py --model tiny --device cpu --epochs 10 --batch-size 32

Explicit CUDA:
python train.py --model small --device cuda --epochs 30 --batch-size 64

## Evaluation

python evaluate.py --checkpoint checkpoints/best.pt

## Smoke test

python smoke_test.py

## Ablation

python experiments/ablation.py

The ablation script is hardware-independent and compares patch size, depth and embedding dimension.

## Reference

Tolstikhin et al., MLP-Mixer: An all-MLP Architecture for Vision.
