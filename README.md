# GULP Project Setup and Usage

This repository contains the implementation and experiments for the GULP activation function paper.

## Project Structure

```
GULP/
├── src/
│   ├── act/
│   │   ├── gulp.py              # GULP activation implementation
│   │   └── gulp_param_sweep.py  # Parameter sweep utilities
│   ├── data/
│   │   └── dataset.py           # Dataset loaders (CIFAR-10/100)
│   └── models/
│       └── model_factory.py     # Model creation with timm
├── docs/
│   ├── main.md                  # Main paper document
│   └── dataset.md               # Dataset specifications
├── train.py                     # Main training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd "c:\Users\hiban\Desktop\code space\GULP"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Test

Before running full experiments, test that all components work:

```bash
# Test individual components
python test_components.py

# Run quick experiment test (2 epochs)
python quick_test.py
```

## Table 3 Experiments

To reproduce Table 3 results from the paper:

### Single Experiment

Run a single experiment with specific parameters:

```bash
python train.py \
    --dataset cifar10 \
    --model resnet18 \
    --activation gulp \
    --epochs 100 \
    --batch_size 128 \
    --seed 42 \
    --gulp_alpha 1.2 \
    --gulp_amp 0.25 \
    --gulp_mu 1.0 \
    --gulp_sigma 0.5
```

### Batch Experiments (Table 3)

Run all experiments for Table 3 with multiple seeds:

```bash
# Full experiments (will take several hours)
python run_table3_experiments.py --epochs 100 --seeds 42 123 456 789 999

# Quick test version (for debugging)
python run_table3_experiments.py --epochs 10 --seeds 42 123
```

### Available Arguments

**Training Script (`train.py`)**:

- `--dataset`: Dataset name (`cifar10`, `cifar100`)
- `--model`: Model architecture (any timm model, e.g., `resnet18`, `wide_resnet28_10`)
- `--activation`: Activation function (`relu`, `gelu`, `silu`, `gulp`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--optimizer`: Optimizer (`sgd`, `adam`, `adamw`)
- `--seed`: Random seed
- `--gulp_*`: GULP-specific parameters

**GULP Parameters**:
- `--gulp_alpha`: Gate slope parameter (default: 1.2)
- `--gulp_amp`: Pulse amplitude (default: 0.25)
- `--gulp_mu`: Pulse center (default: 1.0)
- `--gulp_sigma`: Pulse width (default: 0.5)
- `--gulp_n_bumps`: Number of bumps (default: 1)

## Dataset Information

The project automatically downloads datasets:

- **CIFAR-10**: 50,000 training + 10,000 test images (10 classes)
- **CIFAR-100**: 50,000 training + 10,000 test images (100 classes)

Datasets are stored in Parquet format for efficient loading.

## Models

The project uses [timm](https://github.com/rwightman/pytorch-image-models) for model implementations:

- **ResNet-18**: `resnet18`
- **Wide-ResNet-28-10**: `wide_resnet28_10` 
- **Wide-ResNet-101-2**: `wide_resnet101_2`
- **Vision Transformer**: `vit_tiny_patch16_224`, `vit_small_patch16_224`
- **DeiT**: `deit_tiny_patch16_224`, `deit_small_patch16_224`

Models are automatically adapted for CIFAR datasets (32x32 input) vs ImageNet (224x224).

## Results

Experiment results are saved in the `experiments/` directory:

```
experiments/
├── cifar10_resnet18_gulp_20240816_142030_seed42/
│   ├── config.json           # Experiment configuration
│   ├── results.json          # Training metrics
│   ├── best_model.pth        # Best model checkpoint
│   ├── final_model.pth       # Final model checkpoint
│   └── tensorboard/          # TensorBoard logs
├── table3_results.csv        # All experiment results
├── table3_summary.csv        # Aggregated statistics
└── table3_summary.json       # Table 3 formatted results
```

## Table 3 Format

The batch experiment runner automatically generates Table 3:

```
TABLE 3: CIFAR-10/100 accuracy ± std (n=5 seeds)

CIFAR-10:
| Model | ReLU | GELU | SiLU | GULP |
|-------|------|------|------|------|
| resnet18 | 94.23 ± 0.15 | 94.31 ± 0.12 | 94.28 ± 0.18 | 94.45 ± 0.11 |

CIFAR-100:
| Model | ReLU | GELU | SiLU | GULP |
|-------|------|------|------|------|
| wide_resnet28_10 | 81.15 ± 0.22 | 81.33 ± 0.19 | 81.29 ± 0.25 | 81.58 ± 0.17 |
```

## GULP Activation

The GULP activation function is defined as:

```
GULP(x) = x * σ(αx) * (1 + A * exp(-(x-μ)²/(2σ²)))
```

Where:
- `α`: Gate slope parameter
- `A`: Pulse amplitude 
- `μ`: Pulse center
- `σ`: Pulse width

**Default parameters**: α=1.2, A=0.25, μ=1.0, σ=0.5

## Troubleshooting

**Common Issues**:

1. **CUDA out of memory**: Reduce `--batch_size`
2. **Dataset download fails**: Check internet connection, try manual download
3. **Slow training**: Reduce `--num_workers` or use `--no_cuda` for CPU training
4. **Import errors**: Ensure all dependencies are installed via `requirements.txt`

**Performance Tips**:
- Use `--num_workers 0` for debugging to avoid multiprocessing issues
- Start with small epochs (`--epochs 10`) to test setup
- Use GPU if available for faster training

## Citation

If you use this code, please cite:

```bibtex
@article{gulp2024,
  title={GULP: A Smooth Self-Gated Activation with a Localized Pulse},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

# GULP: A Smooth Self‑Gated Activation with a Localized Pulse

## Abstract

Activation functions critically shape the optimization landscape of deep networks. We propose GULP, a smooth, self‑gated activation that augments the Swish/SiLU family with a localized pulse (bump) term. Formally, GULP multiplies a Swish gate with a unimodal Gaussian bump centered near the positive region, yielding a mild non‑monotonic uplift around the activation threshold while retaining Swish‑like tails. This design enhances gradient flow around moderate positive pre‑activations with negligible overhead. GULP is drop‑in compatible with MLP, CNN, and Transformer feed‑forward layers, and can also serve as the gate in GLU‑style blocks. We provide the mathematical formulation, derivatives, and a comprehensive empirical evaluation across vision, language, audio, graphs, and tabular tasks.