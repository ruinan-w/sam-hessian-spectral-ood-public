# Beyond Scalar Flatness: Hessian Spectral Diagnostics for SAM-Style OOD Robustness

This repository contains the public research code and processed results for the ICONIP 2026 submission:

**Beyond Scalar Flatness: Hessian Spectral Diagnostics for SAM-Style OOD Robustness**

The project studies whether SAM-style optimization improves corruption robustness in residual CIFAR classifiers, and whether the effect is better described by Hessian spectral geometry than by a scalar dominant-eigenvalue flatness measure alone. The processed results cover CIFAR-10-C and CIFAR-100-C evaluations for ResNet-18, plus a ResNet-34 CIFAR-10-C validation setting.

## Repository Structure

```text
sam-hessian-spectral-ood-public/
├── configs/      # Reference experiment settings used for the paper
├── docs/         # Reproducibility notes and clean repository audit
├── figures/      # Final paper figures as PDF/PNG
├── results/      # Final cleaned CSV tables used in the paper
├── scripts/      # Training, evaluation, Hessian, analysis, and plotting entry points
├── src/          # Data loaders, models, optimizers, metrics, and Hessian utilities
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

Create a fresh Python environment and install the dependencies:

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build if GPU acceleration is required. See the official PyTorch installation selector for the command matching your CUDA version.

## Data

The scripts expect datasets under `data/` by default. CIFAR-10 and CIFAR-100 are downloaded through `torchvision`. CIFAR-10-C and CIFAR-100-C should be placed under `data/CIFAR-10-C` and `data/CIFAR-100-C`, respectively, using the standard corruption `.npy` layout.

## Train a Model

Example: train ResNet-18 on CIFAR-10 with SAM and seed 42.

```bash
python scripts/train_experiment.py \
  --dataset cifar10 \
  --model resnet18 \
  --optimizer sam \
  --seed 42 \
  --epochs 100 \
  --batch-size 128 \
  --output-dir outputs
```

The paper settings use SGD, SAM, and ASAM with seeds 42, 43, and 44. Reference settings are listed in `configs/core_experiments.yaml`.

## Evaluate CIFAR-C Robustness

Example: evaluate a trained checkpoint on CIFAR-10-C.

```bash
python scripts/eval_cifar_c.py \
  --dataset cifar10 \
  --model resnet18 \
  --checkpoint outputs/checkpoints/example.pt \
  --run-name example \
  --optimizer-name sam \
  --seed 42 \
  --data-root data \
  --output-dir outputs
```

The evaluation averages accuracy over the 15 standard CIFAR-C corruption types and severities 1 through 5 unless overridden.

## Compute Hessian Metrics

Example: compute dominant-eigenvalue, trace, participation-ratio, and optional Lanczos top-k spectral metrics.

```bash
python scripts/compute_hessian_geometry.py \
  --dataset cifar10 \
  --model resnet18 \
  --checkpoint outputs/checkpoints/example.pt \
  --run-name example \
  --optimizer-name sam \
  --seed 42 \
  --data-root data \
  --subset-size 1024 \
  --use-lanczos \
  --top-k 20 \
  --output-dir outputs
```

## Reproduce Tables and Figures

The final processed CSV files used in the paper are in `results/`. The final figure files are in `figures/`.

To regenerate a processed result package from completed experiment outputs, use:

```bash
python scripts/generate_iconip_results_package.py
```

This script expects completed training, CIFAR-C evaluation, and Hessian output files under `outputs/`. The public repository intentionally does not include checkpoints, raw logs, datasets, private manuscript drafts, or supplementary-preparation files.

## Citation

If you use this code or processed results, please cite the paper. A final BibTeX entry will be added after publication.

```bibtex
@misc{wang2026hessiansamood,
  title        = {Beyond Scalar Flatness: Hessian Spectral Diagnostics for SAM-Style OOD Robustness},
  author       = {Wang, Ruinan and Hu, Yaru and Qiao, Di},
  year         = {2026},
  note         = {ICONIP 2026 submission}
}
```

## Release Note

This public release contains code and processed results corresponding to the ICONIP 2026 submission. It is intentionally minimal: checkpoints, datasets, raw logs, intermediate analysis folders, private manuscript drafts, and supplementary-preparation files are excluded.
