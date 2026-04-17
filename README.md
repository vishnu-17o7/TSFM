# TSFM: Feature-Bootstrapped Masked Time-Series Pretraining

Compact, reproducible masked pretraining for time-series transformers.

This repository trains a TSFM-style encoder with masked patch reconstruction using:
- raw `.tsf` / `.ts` files when available, or
- automatic fallback synthesis from `*_features.csv` files when raw archives are absent.

It also includes deterministic checkpoint evaluation and multi-seed ablation scripts used in the paper draft.

## Table of Contents

- [Project Scope](#project-scope)
- [Repository Layout](#repository-layout)
- [Method Overview](#method-overview)
- [Environment Setup](#environment-setup)
- [Data Requirements](#data-requirements)
- [Training](#training)
- [Checkpoint Evaluation](#checkpoint-evaluation)
- [Benchmark Evaluation](#benchmark-evaluation-tsfm-vs-timesfm)
- [Multi-Seed Ablations](#multi-seed-ablations)
- [Outputs and Artifacts](#outputs-and-artifacts)
- [Reproducibility Notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [Project Status and Limitations](#project-status-and-limitations)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Project Scope

The codebase provides an end-to-end workflow for self-supervised TSFM pretraining:
- Data loading from nested directories (`.tsf`, `.ts`) via `sktime`
- Feature-bootstrap synthesis fallback when raw archives are unavailable
- RevIN + patch embedding + Transformer encoder + latent reconstruction head
- Mixed precision training, gradient accumulation, clipping, and optional `torch.compile`
- Checkpoint sweep evaluation with generated CSV/JSON summaries and plots
- Multi-seed ablation runner with aggregate summary statistics

## Repository Layout

Key files and folders:

```text
train_tsfm.py                  # Main masked pretraining script
evaluate_checkpoints.py        # Checkpoint sweep evaluation + plotting
run_multiseed_ablation.py      # Deterministic multi-seed ablation runner
requirements.txt               # Python dependencies
paper_tsfm_ieee.tex            # IEEE paper source
paper_tsfm_draft.md            # Draft manuscript text
checkpoints/                   # Intermediate training checkpoints
experiments/                   # Ablation/evaluation outputs
data/                          # Input datasets (raw and/or feature tables)
tsfm_pretrain.pt               # Final pretrained checkpoint artifact
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
| `--data-dir` | `data` | Root folder for time-series files |
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

### Training outputs

- Intermediate checkpoints: `checkpoints/checkpoint_epoch_<E>_step_<S>.pt`
- Final pretrained checkpoint: `tsfm_pretrain.pt`

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

Run ablations across mask ratio and patch length using multiple seeds.

```powershell
py run_multiseed_ablation.py --data-dir data --seeds 11,42,123 --epochs 1 --max-steps-per-epoch 20 --batch-size 32 --context-length 512 --feature-workers 1 --max-rows-per-feature-file 50
```

Default ablation grid:
- `mask_0p2`, `mask_0p4`, `mask_0p6` (patch length 16)
- `patch_8`, `patch_32` (mask ratio 0.4)

Main outputs:
- Per-run JSON: `experiments/multiseed/*.json`
- Aggregated summary: `experiments/multiseed_summary.csv`

Related study artifacts already present in the repo include:
- `experiments/multiseed_runs.csv`
- `experiments/multiseed_effects.csv`
- `experiments/ablation_summary_det.csv`

## Outputs and Artifacts

Important generated and tracked artifacts:

- Model checkpoints: `checkpoints/` and `tsfm_pretrain.pt`
- Ablation reports: `experiments/ablations`, `experiments/ablations_det`, `experiments/multiseed`
- Checkpoint evaluation reports: `experiments/checkpoint_eval`
- Paper files: `paper_tsfm_ieee.tex`, `paper_tsfm_ieee.pdf`, `paper_tsfm_draft.md`

## Reproducibility Notes

- Use fixed seeds (`--seed` for training, `--seeds` for multi-seed ablation).
- Keep `--feature-workers 1` if you want stricter deterministic row-processing order in fallback mode.
- Keep `--max-rows-per-feature-file` fixed when comparing runs.
- Use the same `context_length`, `patch_length`, and `mask_ratio` when comparing checkpoint quality.
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

## Project Status and Limitations

- The current repository is research-oriented and optimized for reproducibility experiments.
- Feature fallback synthesis is useful for bootstrapping but is not equivalent to raw-series pretraining.
- The objective is masked latent reconstruction, not direct forecasting loss.
- There is no published benchmark leaderboard in this repository yet.

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
