# Reproducibility Notes

## Datasets

The paper uses CIFAR-10, CIFAR-100, CIFAR-10-C, and CIFAR-100-C. CIFAR-10 and CIFAR-100 are loaded through `torchvision`. CIFAR-C datasets are expected in the standard `.npy` corruption format under `data/CIFAR-10-C` and `data/CIFAR-100-C`.

## Architectures

The main residual settings are:

- CIFAR-10-C / ResNet-18
- CIFAR-100-C / ResNet-18
- CIFAR-10-C / ResNet-34

The codebase also contains a VGG helper because the training script supports it, but the paper's central evidence is based on residual CIFAR classifiers.

## Optimizers and Seeds

The processed paper results compare SGD, SAM, and ASAM with seeds 42, 43, and 44.

## Training Budget

The final processed tables correspond to 100-epoch CIFAR training runs with batch size 128, learning rate 0.1, momentum 0.9, weight decay 5e-4, and cosine scheduling. SAM uses rho 0.05; ASAM uses rho 0.5 and eta 0.01 in the released scripts.

## CIFAR-C Evaluation Rule

CIFAR-C robustness is computed by evaluating each checkpoint over the 15 standard corruption types and severities 1 through 5. The reported CIFAR-C accuracy is the mean across the evaluated corruption/severity grid. The OOD gap is clean accuracy minus CIFAR-C accuracy.

## Hessian Metric Computation

Hessian geometry is estimated on a fixed-size training subset. The released code computes the dominant Hessian eigenvalue by power iteration, estimates trace and participation ratio with stochastic probes, and can compute top-k spectral summaries with Lanczos. Processed tables include dominant eigenvalue, trace estimate, top-5 spectral mass ratio, lambda-max-over-top-k mass, effective rank, spectral entropy, and participation ratio.

## Tables and Figures

The `results/` directory contains final cleaned CSV files used for the ICONIP 2026 submission. The `figures/` directory contains final paper figures in PDF and PNG form. Regenerating the full package from raw experiment outputs requires completed training checkpoints, CIFAR-C evaluation logs, and Hessian logs under `outputs/`, which are intentionally excluded from this public release.
