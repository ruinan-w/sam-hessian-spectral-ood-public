# Final PDF Alignment Check

Checked against `iconip (2).pdf`, last modified 2026-05-20 16:02:52.

## Covered workflow

The public repository contains the code paths required by the final paper workflow:

- training entry point: `scripts/train_experiment.py`
- CIFAR-C evaluation: `scripts/eval_cifar_c.py`
- Hessian geometry computation: `scripts/compute_hessian_geometry.py`
- batch runners: `scripts/run_training_batch.py`, `scripts/run_cifarc_batch.py`, `scripts/run_hessian_batch.py`
- model definitions: `src/models/resnet_cifar.py`
- optimizers: `src/optim/sam.py`, `src/optim/asam.py`, plus PyTorch SGD in the training script
- CIFAR and CIFAR-C loaders: `src/data/`
- Hessian utilities: `src/analysis/hessian_geometry.py`
- final cleaned paper tables: `results/`
- final-paper-aligned figures: `figures/final_fig2_*` through `figures/final_fig6_*`

## Final paper settings covered

- ID datasets: CIFAR-10 and CIFAR-100
- OOD datasets: CIFAR-10-C and CIFAR-100-C
- architectures: ResNet-18 and CIFAR-10-C / ResNet-34 validation
- optimizers: SGD, SAM, ASAM
- seeds: 42, 43, 44
- training budget: 100 epochs
- trajectory diagnostic: CIFAR-10-C / ResNet-18 / seed 42, epochs 20, 40, 60, 80, 100

## Notes

Figure 1 in the final PDF is a conceptual pipeline diagram embedded in the manuscript. I did not find a standalone final Figure 1 source file in the project tree. The public repository therefore contains all available final experimental figures and processed result tables, but not a separate Figure 1 asset.

No checkpoints, raw logs, datasets, manuscript TeX, supplementary-preparation files, or private process files are included.
