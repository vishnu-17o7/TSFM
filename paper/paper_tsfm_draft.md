# Feature-Bootstrapped Masked TSFM: Practical Self-Supervised Pretraining from Monash Feature Tables

**Author:** Vishn  
**Date:** 2026-03-30

## Abstract

We present a compact time-series foundation pretraining pipeline that runs from one training script and one data root. The pipeline loads raw `.tsf` and `.ts` files when they exist. When raw files do not exist, it synthesizes pseudo-series from `*_features.csv` tables and trains a masked patch reconstruction model. The model combines RevIN, convolutional patch embedding, a Transformer encoder stack, and an MLP reconstruction head. The training loop uses mixed precision, gradient accumulation, gradient clipping, and optional graph compilation. This design gives practitioners a direct path to build a reusable initialization for downstream forecasting. We ground the paper in the current repository state: `44` feature CSV files, `0` raw TSF files, `0` raw TS files, and one pretrained checkpoint artifact.

## 1. Introduction

Forecasting teams need two things at once: scale and transfer. They need a model that absorbs broad temporal patterns, and they need a training path they can run on local hardware.

TimesFM [1] shows that a decoder-only transformer pretrained on large and diverse corpora can deliver strong zero-shot forecasting across unseen datasets. That result sets a clear direction: use patch-based tokenization, train on broad temporal variation, and target transfer.

This repository follows that direction but takes a different training objective and deployment path. We train a masked patch reconstruction model in PyTorch with an encoder backbone. We also include a feature-driven fallback loader that converts statistical feature tables into pseudo-series, which keeps pretraining possible when raw series files are absent.

This paper makes three concrete contributions.

1. We define a code-complete TSFM pretraining recipe that runs from one script: [train_tsfm.py](train_tsfm.py).
2. We introduce a feature-bootstrap data path that maps tabular time-series descriptors into synthetic sequences for self-supervised pretraining.
3. We document a reproducible training protocol and artifact layout for fast downstream reuse.

## 2. Relation to TimesFM

TimesFM [1] and this work share one core idea: patch tokens give transformers a scalable interface for time-series pretraining.

The methods diverge in three places.

1. **Backbone direction:** TimesFM uses a decoder-only causal stack for next-patch forecasting. This repository uses an encoder stack for masked patch reconstruction.
2. **Training target:** TimesFM predicts future horizons in autoregressive style with long output patches. This repository reconstructs masked latent patch embeddings from the same context window.
3. **Data bootstrap path:** TimesFM relies on a large real plus synthetic corpus at internet scale. This repository can train from local raw series files, then fall back to feature-table synthesis when raw files are missing.

These differences trade large-scale zero-shot forecasting for a lighter, code-first pretraining path that users can run and inspect end to end.

## 3. Data Pipeline

### 3.1 Input discovery

The loader scans the data root in recursive mode.

1. It first looks for raw files: `.tsf` and `.ts`.
2. It loads `.tsf` through `load_tsf_to_dataframe` candidates from `sktime`.
3. It loads `.ts` through `load_from_tsfile_to_dataframe` candidates from `sktime`.
4. It keeps finite numeric values and drops invalid entries.

### 3.2 Feature fallback synthesis

When the loader finds no raw files, the script can synthesize pseudo-series from `*_features.csv` rows.

Each row defines shape controls such as `mean`, `var`, `x_acf1`, `trend`, `seasonal_period`, and `spike`. The generator constructs a sequence by summing trend, seasonal, noise, random-walk, and spike components, then rescales to the requested mean and variance. The script supports `polars` and `pandas` backends, row caps per file, and thread-level parallel file processing.

### 3.3 Repository data state

The current workspace exposes this inventory.

1. Raw TSF files: `0`
2. Raw TS files: `0`
3. Feature CSV files matching `*_features.csv`: `44`

Given that state, pretraining in this repository flows through the feature fallback path.

## 4. Model

### 4.1 Windowing

The dataset class samples sliding windows of length `context_length` and step `stride` from each series.

Given a series $s \in \mathbb{R}^{T}$, the loader creates windows

$$
w_k = s[k \cdot \text{stride} : k \cdot \text{stride} + L], \quad L=\text{context\_length}.
$$

The model receives each window as a univariate tensor with shape $(L, 1)$.

### 4.2 RevIN normalization

The model applies RevIN to each sample window before tokenization. RevIN stores sample mean and variance, then applies learnable affine scaling.

### 4.3 Patch embedding

The patch layer uses `Conv1d` with kernel size and stride equal to `patch_length`. This operation maps each non-overlapping patch to one token in embedding space. A learned positional embedding adds token order.

### 4.4 Transformer backbone and reconstruction head

The encoder stack uses `num_layers` TransformerEncoderLayer blocks with GELU activation, `norm_first=True`, and multi-head attention. A learned `mask_token` replaces masked token vectors. The head maps encoded tokens through a two-layer MLP.

With default arguments (`context_length=512`, `patch_length=16`, `embed_dim=128`, `num_layers=4`), the model holds about $8.33 \times 10^5$ trainable parameters.

## 5. Self-Supervised Objective

The script samples a Bernoulli mask per patch with probability `mask_ratio`. It replaces masked patch tokens with the learned mask token, encodes the sequence, and reconstructs patch embeddings.

Let $E \in \mathbb{R}^{N \times d}$ denote clean patch embeddings, $M \in \{0,1\}^{N}$ denote mask indicators, and $\hat{E}$ denote predicted embeddings. The model minimizes

$$
\mathcal{L}_{\text{mask}} = \frac{1}{|\{i: M_i=1\}|} \sum_{i: M_i=1} \|\hat{E}_i - E_i\|_2^2.
$$

This target trains the encoder to recover latent patch content from partial context.

## 6. Training System

The training loop in [train_tsfm.py](train_tsfm.py) uses the following controls.

1. Optimizer: AdamW with configurable learning rate and weight decay.
2. Stability: gradient norm clipping at `1.0`.
3. Throughput: optional AMP on CUDA and optional `torch.compile`.
4. Effective batch scaling: gradient accumulation across `gradient_accumulation_steps` micro-batches.
5. Data throughput: configurable DataLoader workers, pinned memory on CUDA, persistent workers and prefetch when worker count is positive.

The script saves a final checkpoint to `tsfm_pretrain.pt` with model weights and full argument state.

## 7. Artifact and Reproducibility

This repository already includes one checkpoint artifact.

1. File: `tsfm_pretrain.pt`
2. Size: `3,352,412` bytes
3. Last write time: `2026-02-20 15:33:01`

You can reproduce a short smoke run with:

```powershell
py train_tsfm.py --data-dir data --epochs 1 --max-steps-per-epoch 1
```

You can run a longer pretraining run with:

```powershell
py train_tsfm.py --data-dir data --epochs 10 --context-length 512 --patch-length 16 --batch-size 32 --mask-ratio 0.4
```

When you use feature fallback at scale, you can cap rows and use thread workers:

```powershell
py train_tsfm.py --data-dir data --feature-workers 8 --max-rows-per-feature-file 5000 --progress-every-rows 2000
```

## 8. Discussion

This codebase gives a practical route to TSFM pretraining. The path favors transparency, local control, and low setup cost.

The same design introduces known limits.

1. The current objective reconstructs latent patches, not future horizons.
2. The fallback generator depends on feature engineering quality and can miss sequence motifs that raw sensors capture.
3. The repository lacks benchmark scripts that report zero-shot forecasting metrics against supervised baselines.

Teams can address those limits by adding a forecasting fine-tune stage, benchmark harnesses for Monash and ETT tasks, and ablations over mask ratio, patch length, and synthetic length.

## 9. Conclusion

We presented a compact TSFM pretraining recipe that users can run from one script and one data root. The method uses robust file discovery, feature-table synthesis, masked patch reconstruction, and hardware-aware optimization. The result is a reusable pretraining artifact that can seed downstream forecasting research in settings where raw series access stays limited.

## References

[1] Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou. *A Decoder-only Foundation Model for Time-Series Forecasting*. arXiv:2310.10688v4, 2024.

[2] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. ICLR, 2023.

[3] Taejun Kim, Jinheon Baek, Hwanjun Song, Yunjin Choi, and others. *RevIN: Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*. ICLR, 2022.