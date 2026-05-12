# TSFM: Feature-Bootstrapped Masked Time-Series Pretraining

Compact, reproducible masked pretraining for time-series transformers with world-class rigour.

This repository trains a TSFM-style encoder with masked patch reconstruction using:
- raw `.tsf` / `.ts` files when available,
- real-world corpora from the Monash Forecasting Repository and M4 Competition, or
- automatic fallback synthesis from `*_features.csv` files when raw archives are absent.

It includes deterministic checkpoint evaluation, multi-seed ablation scripts with bootstrap confidence intervals and effect-size reporting, and seed-count planning tools — all designed for journal-ready statistical power.

## Table of Contents

- [Project Scope](#project-scope)
- [Repository Layout](#repository-layout)
- [Method Overview](#method-overview)
- [Environment Setup](#environment-setup)
- [World-Class Quick Start](#world-class-quick-start)
- [Data Requirements](#data-requirements)
- [Training](#training)
- [Checkpoint Evaluation](#checkpoint-evaluation)
- [Benchmark Evaluation](#benchmark-evaluation-tsfm-vs-timesfm)
- [Multi-Seed Ablations](#multi-seed-ablations)
- [Training Curves Visualization](#training-curves-visualization)
- [Seed-Count Planning](#seed-count-planning)
- [Outputs and Artifacts](#outputs-and-artifacts)
- [Reproducibility Notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Project Scope

The codebase provides an end-to-end workflow for self-supervised TSFM pretraining:
- Data loading from nested directories (`.tsf`, `.ts`) via `sktime`
- Real-world corpus acquisition from Monash Forecasting Repository and M4 Competition
- Feature-bootstrap synthesis fallback when raw archives are unavailable
- RevIN + patch embedding + Transformer encoder + latent reconstruction head
- Mixed precision training, gradient accumulation, clipping, and optional `torch.compile`
- Early stopping with validation-based best-model checkpointing
- Checkpoint sweep evaluation on the full 7/7 TimesFM benchmark suite
- Multi-seed ablation runner with bootstrap CIs, Cohen's d effect sizes, and seed-count planning
- Publication-quality training curve plotting

## Repository Layout

Key files and folders:

```text
train_tsfm.py                  # Main masked pretraining script
finetune_forecasting.py        # Downstream fine-tuning pipeline
evaluate_checkpoints.py        # Checkpoint sweep + benchmark evaluator (TSFM vs TimesFM)
evaluate_results.py            # Results analyzer and report generator
run_multiseed_ablation.py      # Multi-seed ablation runner with bootstrap CIs and effect sizes
download_pretraining_corpora.py # Real-world corpus downloader (Monash + M4)
prepare_datasets.py             # Dataset download and preparation utility
benchmark_leaderboard.py       # Ranked benchmark leaderboard printer
validate_setup.py              # Dependency and setup validation
config_example.ini             # Configuration template
requirements.txt               # Python dependencies
checkpoints/                   # Intermediate training checkpoints
experiments/                   # Ablation/evaluation outputs
  training_curves.py           # Publication-quality loss curve plotting
  seed_count_planner.py        # Statistical power analysis tool
data/                          # Input datasets (raw and/or feature tables)
  real_corpora/                # Downloaded real-world .tsf files (Monash, M4)
tsfm_pretrain.pt               # Final pretrained checkpoint artifact
tsfm_best.pt                   # Best validation-loss model checkpoint
```

## Method Overview

1. Build sliding windows from each series (`context_length`, `stride`).
2. Normalize each sample with RevIN.
3. Convert windows to patch tokens using `Conv1d` patchification.
4. Randomly mask patch tokens using `mask_ratio`.
5. Encode with a Transformer encoder stack.
6. Reconstruct latent patch embeddings and optimize masked MSE.

Core model components in `train_tsfm.py`:
- `RevIN`
- `PatchEmbedding`
- `TSTransformerBackbone`
- `TSFMPretrain`
- `WindowDataset`

## Environment Setup

### 1) Create and activate a virtual environment (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 3) Optional packages

`evaluate_checkpoints.py` uses matplotlib for plots. If not already installed:

```powershell
py -m pip install matplotlib
```

## World-Class Quick Start

Complete pipeline — download real corpora, pretrain with early stopping, run 7-seed ablations with bootstrap CIs, and evaluate on all 7 benchmark datasets:

```powershell
# 1. Download real-world pretraining corpora (Monash + M4, ~38 datasets)
py download_pretraining_corpora.py --output-dir data/real_corpora --max-datasets 10

# 2. Prepare the full 7/7 benchmark suite
py prepare_datasets.py --timesfm-benchmarks

# 3. Pretrain with real corpora, early stopping, and best-model saving
py train_tsfm.py --data-dir data --real-data-dir data/real_corpora --epochs 10 --early-stopping-patience 5 --best-model-path tsfm_best.pt --metrics-out experiments/train_metrics.json

# 4. Run multi-seed ablations with 7 seeds, bootstrap CIs, and convergence tracking
py run_multiseed_ablation.py --data-dir data --seeds 11,42,123,256,512,789,1024 --epochs 10 --ci-method both

# 5. Visualize training curves
py experiments/training_curves.py --metrics-dir experiments/multiseed --output experiments/training_curves.png

# 6. Estimate required seeds for target precision
py experiments/seed_count_planner.py --pilot-csv experiments/multiseed_summary.csv

# 7. Evaluate on the full 7/7 benchmark suite
py evaluate_checkpoints.py --models both --checkpoint-glob "tsfm_best.pt" --forecast-horizon 96 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Electricity Traffic Weather

# 8. Print the ranked leaderboard
py benchmark_leaderboard.py --input finetuning_results/benchmark_results.json --sort-by mean_mse --show-dataset-winners
```

**What this gives you:**
- Real-world pretraining data (not just synthetic) with tracked data composition
- Early stopping + best-model checkpointing for convergence-guaranteed training
- 7-seed ablations with bootstrap confidence intervals and Cohen's d effect sizes
- Seed-count planner for statistical power analysis
- Training curve visualization for convergence diagnostics
- Full 7/7 benchmark evaluation (Electricity and Weather included)

## Data Requirements

Point `--data-dir` to the dataset root (default: `data`). Loader behavior:

- First pass: recursively load raw `.tsf` and `.ts` files.
- Fallback pass (enabled by default): if no raw files are found, scan for `*_features.csv` and synthesize pseudo-series.

### Supported input patterns

- `**/*.tsf`
- `**/*.ts`
- `**/*_features.csv` (fallback mode)

### Disable fallback (strict raw-only mode)

```powershell
py train_tsfm.py --data-dir data --no-feature-fallback
```

### Real-world pretraining corpora

Download real time-series archives from the Monash Forecasting Repository (~38 datasets) and M4 Competition:

```powershell
# All Monash + M4 datasets (~38 corpora)
py download_pretraining_corpora.py --output-dir data/real_corpora

# Specific datasets only
py download_pretraining_corpora.py --output-dir data/real_corpora --datasets m4_hourly m4_daily electricity_hourly

# First 10 datasets only (faster download)
py download_pretraining_corpora.py --output-dir data/real_corpora --max-datasets 10
```

Train with both synthetic and real data:

```powershell
py train_tsfm.py --data-dir data --real-data-dir data/real_corpora --epochs 10
```

Train with real data only (no synthetic fallback):

```powershell
py train_tsfm.py --real-data-dir data/real_corpora --real-data-only --epochs 10
```

The data composition (synthetic vs real series count and ratio) is tracked in metrics output and ablation summaries.

## Training

All commands are run from repository root.

### Smoke test

```powershell
py train_tsfm.py --data-dir data --epochs 1 --max-steps-per-epoch 1
```

### Standard pretraining run

```powershell
py train_tsfm.py --data-dir data --epochs 10 --context-length 512 --patch-length 16 --batch-size 32 --mask-ratio 0.4
```

### Pretraining with early stopping and best-model saving

```powershell
py train_tsfm.py --data-dir data --epochs 10 --early-stopping-patience 5 --early-stopping-min-delta 1e-5 --best-model-path tsfm_best.pt --metrics-out experiments/train_metrics.json
```

### Pretraining with real-world corpora

```powershell
py train_tsfm.py --data-dir data --real-data-dir data/real_corpora --epochs 10 --best-model-path tsfm_best.pt
```

### Faster feature-fallback preprocessing

```powershell
py train_tsfm.py --data-dir data --feature-workers 8 --max-rows-per-feature-file 5000 --progress-every-rows 2000
```

### CPU-safe / lower-memory variant

```powershell
py train_tsfm.py --data-dir data --num-workers 4 --feature-workers 4 --no-amp --no-compile --batch-size 16
```

### Main training flags

| Flag | Default | Purpose |
|---|---:|---|
| `--data-dir` | `data` | Root folder for synthetic/feature time-series files |
| `--real-data-dir` | `None` | Directory with real `.tsf`/`.ts` pretraining corpora |
| `--real-data-only` | off | Skip synthetic fallback; require real corpora |
| `--epochs` | `10` | Number of epochs |
| `--batch-size` | `32` | Batch size |
| `--context-length` | `512` | Window length |
| `--patch-length` | `16` | Patch/token length (must divide context length) |
| `--embed-dim` | `128` | Token embedding width |
| `--num-heads` | `8` | Attention heads |
| `--num-layers` | `4` | Transformer layers |
| `--dropout` | `0.1` | Dropout rate |
| `--mask-ratio` | `0.5` | Bernoulli masking probability |
| `--stride` | `16` | Sliding-window stride |
| `--lr` | `1e-4` | AdamW learning rate |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--save-every` | `5000` | Save periodic checkpoints every N global steps |
| `--early-stopping-patience` | `5` | Stop if val loss does not improve for N epochs (0 = disabled) |
| `--early-stopping-min-delta` | `1e-5` | Minimum improvement in val loss to reset patience |
| `--best-model-path` | `tsfm_best.pt` | Path to save the best-checkpoint-by-val-loss model |
| `--metrics-out` | `None` | Path to write per-epoch metrics JSON |
| `--run-name` | `""` | Name for this run (used in metrics output) |
| `--resume-from` | `None` | Path to checkpoint to resume training from |
| `--num-workers` | `-1` | DataLoader workers (`-1` = all CPU cores) |
| `--seed` | `42` | Random seed |
| `--max-steps-per-epoch` | `0` | Cap batches per epoch (`0` = full epoch) |
| `--feature-fallback/--no-feature-fallback` | enabled | Toggle feature CSV fallback |
| `--synthetic-length` | `1024` | Length of synthetic series (fallback mode) |
| `--progress-every-rows` | `5000` | Feature synthesis progress print interval |
| `--feature-workers` | `-1` | Feature CSV worker threads (`-1` = all CPU cores) |
| `--max-rows-per-feature-file` | `0` | Row cap per feature CSV (`0` = all rows) |
| `--gradient-accumulation-steps` | `4` | Gradient accumulation factor |
| `--no-amp` | off | Disable mixed precision |
| `--no-compile` | off | Disable `torch.compile` |
| `--loss-fn` | `huber` | Loss function (`mse` or `huber`) |
| `--gradient-checkpointing` | off | Enable gradient checkpointing for memory savings |
| `--ema-decay` | `0.999` | EMA decay rate for weight averaging (`0` = disabled) |
| `--augment / --no-augment`| enabled | Toggle time-series data augmentation |

### Training outputs

- Intermediate checkpoints: `checkpoints/checkpoint_epoch_<E>_step_<S>.pt`
- Best validation-loss checkpoint: `tsfm_best.pt` (when `--best-model-path` is set)
- Final pretrained checkpoint: `tsfm_pretrain.pt`
- Per-epoch metrics JSON: `experiments/train_metrics.json` (when `--metrics-out` is set)

## Checkpoint Evaluation

Evaluate a set of checkpoints on a deterministic validation subset and export metrics + plots.

```powershell
py evaluate_checkpoints.py --data-dir data --checkpoints-dir checkpoints --output-dir experiments/checkpoint_eval --batch-size 64 --eval-mask-ratio 0.5 --max-eval-windows 8192
```

### Useful evaluation flags

| Flag | Default | Purpose |
|---|---:|---|
| `--checkpoint-pattern` | `checkpoint_epoch_*_step_*.pt` | Glob for checkpoints |
| `--eval-mask-ratio` | `0.5` | Mask ratio used at eval |
| `--val-ratio` | `0.05` | Validation split ratio |
| `--max-eval-windows` | `8192` | Cap eval windows |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--reuse-csv` | `False` | Skip reevaluation and regenerate plots from existing CSV |

Generated files in `experiments/checkpoint_eval`:
- `final_checkpoint_eval.csv`
- `final_checkpoint_eval_summary.json`
- `checkpoint_mse_trend.png`
- `checkpoint_speed_quality.png`

## Benchmark Evaluation (TSFM vs TimesFM)

Use the evaluator to compare TSFM checkpoints with a pip-installed TimesFM baseline on:

- ETTh1
- ETTh2
- ETTm1
- ETTm2
- Electricity
- Traffic
- Weather

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Prepare benchmark datasets:

```powershell
py prepare_datasets.py --timesfm-benchmarks
```

Run evaluation:

```powershell
py evaluate_checkpoints.py --models both --checkpoint-glob "finetuning_results/model_*.pt" --forecast-horizon 96
```

Notes:

- `checkpoints/*.pt` are pretraining checkpoints and typically miss `forecasting_head.*`; evaluator now fails fast on those by default.
- If a TSFM checkpoint horizon is shorter than `--forecast-horizon` (for example 24 vs 96), evaluator uses autoregressive rollout to match the requested horizon.

Output is saved to:

- `finetuning_results/benchmark_results.json`

Print a ranked leaderboard from the saved benchmark JSON:

```powershell
py benchmark_leaderboard.py --input finetuning_results/benchmark_results.json --sort-by mean_mse --show-dataset-winners
```

### Scaling Reconciliation Rules

The evaluator enforces scale-safe comparison between TSFM and TimesFM:

1. TimesFM uses internal normalization:
- v2.5 API: `ForecastConfig(normalize_inputs=True)`
- v1 API: `forecast(..., normalize=True)`
2. TSFM inputs are z-normalized with train-split stats by default, then TSFM outputs are inverse-scaled back to raw units.
3. MSE/MAE are always computed in raw dataset units.

Positive examples:

- TimesFM normalize enabled, forecast compared directly to raw target.
- TSFM context normalized by train mean/std, then prediction inverse-scaled with the same stats before metric calculation.

Negative examples:

- Enabling TimesFM normalization and then manually inverse-scaling TimesFM outputs (double inverse).
- Manually z-scoring TimesFM inputs while also `normalize_inputs=True` (double normalization).
- Comparing normalized TSFM outputs directly against raw targets.

## Multi-Seed Ablations

Run ablations across mask ratio and patch length with statistical rigour: 7 seeds default, bootstrap CIs, Cohen's d effect sizes, validation-based early stopping, and convergence tracking.

### Standard ablation run (7 seeds, full convergence)

```powershell
py run_multiseed_ablation.py --data-dir data --seeds 11,42,123,256,512,789,1024 --epochs 10 --ci-method both
```

### Quick pilot + seed-count estimation

```powershell
py run_multiseed_ablation.py --data-dir data --seeds 11,42,123 --epochs 3 --estimate-seeds --target-ci-half-width-ratio 0.10
```

### Ablation with real-world corpora

```powershell
py run_multiseed_ablation.py --data-dir data --real-data-dir data/real_corpora --seeds 11,42,123,256,512,789,1024 --epochs 10
```

### Key ablation flags

| Flag | Default | Purpose |
|---|---:|---|
| `--seeds` | `11,42,123,256,512,789,1024` | Comma-separated seeds (default: 7) |
| `--epochs` | `10` | Training epochs per run |
| `--max-steps-per-epoch` | `0` | Cap batches per epoch (`0` = full pass) |
| `--early-stopping-patience` | `3` | Stop if val loss stalls for N epochs |
| `--early-stopping-min-delta` | `1e-5` | Min improvement to count as progress |
| `--val-split` | `0.1` | Fraction of data for validation |
| `--ci-method` | `both` | Confidence interval: `t`, `bootstrap`, or `both` |
| `--bootstrap-samples` | `10000` | Bootstrap resamples for CI estimation |
| `--estimate-seeds` | off | Pilot mode: run first 3 seeds only, print required n |
| `--target-ci-half-width-ratio` | `0.10` | Target CI half-width as fraction of mean |
| `--save-checkpoints` | on | Save best-per-run model checkpoints |
| `--data-dir` | `data` | Synthetic/feature data directory |
| `--real-data-dir` | `None` | Real corpora directory |
| `--real-data-only` | off | Skip synthetic fallback |
| `--loss-fn` | `huber` | Loss function (`mse` or `huber`) |
| `--ema-decay` | `0.999` | EMA decay rate for weight averaging (`0` = disabled) |
| `--no-compile` | off | Disable `torch.compile` (PyTorch 2.0+) |
| `--gradient-checkpointing` | off | Enable gradient checkpointing for memory savings |

Default ablation grid:
- `mask_0p2`, `mask_0p4`, `mask_0p6` (patch length 16)
- `patch_8`, `patch_32` (mask ratio 0.4)

Main outputs:
- Per-run JSON: `experiments/multiseed/*.json` (with per-epoch train+val losses and convergence diagnostics)
- Best checkpoints: `experiments/multiseed/checkpoints/*.pt`
- Aggregated summary CSV: `experiments/multiseed_summary.csv`
  - Columns include: `train_mse_mean`, `train_mse_std`, `train_mse_sem`, `train_mse_ci95_lower/upper`, `ci_method`, `significant_at_p05`, `cohens_d_vs_default`, `val_mse_mean`, `converged_fraction`
- Seed-count plan CSV: `experiments/multiseed/seed_count_plan.csv` (when `--estimate-seeds` is used)

## Training Curves Visualization

Plot publication-quality training curves from multi-seed ablation metrics:

```powershell
py experiments/training_curves.py --metrics-dir experiments/multiseed --output experiments/training_curves.png
```

The output shows per-configuration mean train and validation loss curves with std bands across seeds.

## Seed-Count Planning

Estimate the number of seeds needed to achieve a target confidence interval half-width:

```powershell
py experiments/seed_count_planner.py --pilot-csv experiments/multiseed_summary.csv --target-half-width-ratio 0.10 --output-csv experiments/seed_plan.csv
```

The planner uses the pilot's observed variance to compute `n_required = (t_critical * sigma / desired_half_width)^2` for each config and prints how many additional seeds are needed.

## Outputs and Artifacts

Important generated and tracked artifacts:

- Model checkpoints: `checkpoints/`, `tsfm_pretrain.pt`, `tsfm_best.pt`
- Ablation reports: `experiments/multiseed/` (JSON per-run, CSV summary, checkpoints)
- Training curves: `experiments/training_curves.png`
- Seed-count plans: `experiments/seed_count_plan.csv`
- Checkpoint evaluation reports: `experiments/checkpoint_eval/`
- Benchmark results: `finetuning_results/benchmark_results.json`
- Fine-tuning results: `finetuning_results/results.json`
- Real corpora manifest: `data/real_corpora/manifest.json`

## Reproducibility Notes

- Use fixed seeds (`--seed` for training, `--seeds` for multi-seed ablation).
- Keep `--feature-workers 1` if you want stricter deterministic row-processing order in fallback mode.
- Keep `--max-rows-per-feature-file` fixed when comparing runs.
- Use the same `context_length`, `patch_length`, and `mask_ratio` when comparing checkpoint quality.
- When comparing across data sources, track the `real_ratio` in metrics output — different data compositions can shift loss baselines.
- Use `--best-model-path` to always restore the best validation-loss checkpoint before downstream evaluation.
- For publication-grade CIs, use `--ci-method both` and verify that `n_seeds` meets the target via `--estimate-seeds` or `seed_count_planner.py`.
- For paper compilation:

```powershell
pdflatex -interaction=nonstopmode paper_tsfm_ieee.tex
pdflatex -interaction=nonstopmode paper_tsfm_ieee.tex
```

## Troubleshooting

### `Import "polars" could not be resolved` in editor

Install dependencies in the active interpreter and ensure VS Code uses the same virtual environment:

```powershell
py -m pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'matplotlib'`

Install matplotlib for evaluation plotting:

```powershell
py -m pip install matplotlib
```

### `ValueError: context_length must be divisible by patch_length`

Pick compatible values (for example, `context_length=512` with `patch_length=8`, `16`, or `32`).

### CUDA out-of-memory

Reduce memory pressure by lowering `--batch-size`, increasing `--patch-length`, disabling AMP (`--no-amp`), or using CPU mode in evaluation.

### No raw files found

If you want strict raw-series pretraining, provide `.tsf`/`.ts` files and pass `--no-feature-fallback`.
If feature fallback is acceptable, keep fallback enabled and ensure `*_features.csv` files exist under `data`.

## Project Status

The repository has been upgraded to address prior research limitations:

| Limitation | Resolution |
|---|---|
| **Training budget**: 1 epoch / 20 steps was insufficient | Default is now 10 epochs with full dataset passes, validation-based early stopping, and best-model checkpointing |
| **Benchmark coverage**: Only 5/7 datasets evaluated | Electricity and Weather datasets are now included via download mirrors and synthetic fallbacks; 7/7 evaluation completes reliably |
| **Statistical power**: n=3 seeds with wide CIs | Default is now 7 seeds with both Student-t and 10,000-sample bootstrap CIs, Cohen's d effect sizes, significance flags, and seed-count planning tools |
| **Synthetic-only pretraining**: No real-world corpora | Real-world corpora from the Monash Forecasting Repository (~38 datasets) and M4 Competition can be downloaded and used alongside or in place of synthetic data; data composition is tracked in all metrics |

Ongoing considerations:
- The masked latent reconstruction objective is not a direct forecasting loss; downstream benefits depend on the alignment between the pretraining task and the target task.
- The repo remains research-oriented but now provides the statistical power and data diversity expected of a journal-ready paper.

## Contributing

Contributions are welcome.

1. Open an issue describing the bug, experiment request, or feature proposal.
2. Create a branch and keep changes focused (for example: training, evaluation, docs, or paper).
3. Include reproducible command lines and generated artifact paths when reporting results.
4. Submit a pull request with a concise summary and any relevant before/after metrics.

## Citation

If you use this code or artifacts, cite the project paper source in this repository.

```bibtex
@misc{vishn2026featurebootstrappedtsfm,
title        = {Feature-Bootstrapped Masked TSFM: Theory, Complexity, and Controlled Ablations for Reproducible Time-Series Pretraining},
author       = {Vishn},
year         = {2026},
howpublished = {GitHub repository manuscript},
note         = {Source file: paper_tsfm_ieee.tex}
}
```

You can also cite TimesFM for broader foundation-model context:

```bibtex
@article{das2024timesfm,
title   = {A Decoder-only Foundation Model for Time-Series Forecasting},
author  = {Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen},
journal = {arXiv preprint arXiv:2310.10688},
year    = {2024}
}
```

## License

This repository is licensed under the MIT License.
See [LICENSE](LICENSE) for the full text.
