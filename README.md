# TSFM Pretraining (sktime + Monash files)

## Run

```powershell
py -m pip install -r requirements.txt
py train_tsfm.py --data-dir data --epochs 10
```

## Expected input files

The loader recursively scans `data` for raw time-series files:

- `*.tsf` (Monash archive format)
- `*.ts` (sktime time-series format)

If no raw `.tsf`/`.ts` files are present, the script automatically falls back to `*_features.csv` files and synthesizes pseudo-series for pretraining.
This is useful for bootstrapping but is not equivalent to true raw-series pretraining.

## Useful flags

```powershell
py train_tsfm.py --data-dir data --context-length 512 --patch-length 16 --batch-size 32 --mask-ratio 0.4
```

Faster feature-fallback loading (polars + parallel workers + row cap):

```powershell
py train_tsfm.py --data-dir data --feature-workers 8 --max-rows-per-feature-file 5000 --progress-every-rows 2000
```

Disable fallback (require raw files only):

```powershell
py train_tsfm.py --data-dir data --no-feature-fallback
```

For smoke tests:

```powershell
py train_tsfm.py --data-dir data --epochs 1 --max-steps-per-epoch 1
```

## Benchmark Evaluation (TSFM vs TimesFM)

Use the new evaluator to compare TSFM checkpoints with a pip-installed TimesFM baseline on:

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
