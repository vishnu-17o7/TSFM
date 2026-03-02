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
