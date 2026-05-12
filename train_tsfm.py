import argparse
import copy
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import polars as pl

    HAS_POLARS = True
except Exception:
    pl = None
    HAS_POLARS = False


class RevIN(nn.Module):
    def __init__(self, num_features: int = 1, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self._mean = None
        self._var = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._var = x.var(dim=1, keepdim=True, unbiased=False).detach()
            x = (x - self._mean) / torch.sqrt(self._var + self.eps)
            x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self._mean is None or self._var is None:
                raise RuntimeError("RevIN denorm called before norm.")
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * torch.sqrt(self._var + self.eps) + self._mean
            return x
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode}")


class PatchEmbedding(nn.Module):
    def __init__(self, context_length: int, patch_length: int, embed_dim: int) -> None:
        super().__init__()
        if context_length % patch_length != 0:
            raise ValueError("context_length must be divisible by patch_length")
        self.context_length = context_length
        self.patch_length = patch_length
        self.num_patches = context_length // patch_length

        self.value_embedding = nn.Conv1d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_length,
            stride=patch_length,
        )
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.value_embedding(x)
        x = x.transpose(1, 2)
        x = x + self.position_embedding
        return x


class TSTransformerBackbone(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TSFMPretrain(nn.Module):
    def __init__(
        self,
        context_length: int,
        patch_length: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.patch_length = patch_length
        self.gradient_checkpointing = gradient_checkpointing
        self.revin = RevIN(num_features=1)
        self.patch_embedding = PatchEmbedding(context_length, patch_length, embed_dim)
        self.transformer = TSTransformerBackbone(embed_dim, num_heads, num_layers, dropout=dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Reconstruct raw patch values (not embeddings) for better pretraining signal
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_length),
        )

    def _apply_mask(self, embedded_patches: torch.Tensor, mask_ratio: float):
        batch_size, num_patches, embed_dim = embedded_patches.shape

        rand = torch.rand(batch_size, num_patches, device=embedded_patches.device)
        mask = rand < mask_ratio  # (B, N)
        mask_token = self.mask_token.expand(batch_size, num_patches, embed_dim)
        masked = torch.where(mask.unsqueeze(-1), mask_token, embedded_patches)

        return masked, mask

    def forward(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = self.revin(x, mode="norm")
        x_series = x_norm.squeeze(-1)  # (B, context_length)

        # Extract raw patches as the reconstruction target
        B = x_series.shape[0]
        num_patches = self.patch_embedding.num_patches
        raw_patches = x_series.reshape(B, num_patches, self.patch_length)  # (B, N, P)

        embedded = self.patch_embedding(x_series)
        masked, mask = self._apply_mask(embedded, mask_ratio)

        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            encoded = checkpoint(self.transformer, masked, use_reentrant=False)
        else:
            encoded = self.transformer(masked)

        reconstructed = self.reconstruction_head(encoded)  # (B, N, patch_length)
        return reconstructed, raw_patches, mask


class WindowDataset(Dataset):
    def __init__(
        self,
        series_list: Sequence[np.ndarray],
        context_length: int,
        stride: int,
        augment: bool = False,
        aug_scale_range: Tuple[float, float] = (0.8, 1.2),
        aug_jitter_std: float = 0.01,
        aug_cutout_ratio: float = 0.05,
    ) -> None:
        self.series_list = [s.astype(np.float32, copy=False) for s in series_list if len(s) >= context_length]
        if not self.series_list:
            raise ValueError(
                "No series are at least as long as context_length. "
                "Use a smaller context_length or provide longer series."
            )
        self.context_length = context_length
        self.stride = stride
        self.augment = augment
        self.aug_scale_range = aug_scale_range
        self.aug_jitter_std = aug_jitter_std
        self.aug_cutout_ratio = aug_cutout_ratio

        self.windows_per_series = [((len(s) - context_length) // stride) + 1 for s in self.series_list]
        self.cumulative = np.cumsum(self.windows_per_series)
        self.total_windows = int(self.cumulative[-1])

    def __len__(self) -> int:
        return self.total_windows

    def _apply_augmentation(self, window: np.ndarray) -> np.ndarray:
        """Apply random augmentations: scaling, jittering, and cutout."""
        w = window.copy()
        # Random scaling
        scale = np.random.uniform(self.aug_scale_range[0], self.aug_scale_range[1])
        w = w * scale
        # Additive jitter
        std = float(np.std(w))
        if std > 1e-6:
            w = w + np.random.normal(0, self.aug_jitter_std * std, size=w.shape).astype(np.float32)
        # Random cutout (zero out a contiguous segment)
        cutout_len = int(len(w) * self.aug_cutout_ratio)
        if cutout_len > 0:
            start = np.random.randint(0, len(w) - cutout_len + 1)
            w[start : start + cutout_len] = 0.0
        return w

    def __getitem__(self, idx: int) -> torch.Tensor:
        series_idx = int(np.searchsorted(self.cumulative, idx, side="right"))
        prior = 0 if series_idx == 0 else int(self.cumulative[series_idx - 1])
        local_window_idx = idx - prior
        start = local_window_idx * self.stride
        series = self.series_list[series_idx]
        window = series[start : start + self.context_length]
        if self.augment:
            window = self._apply_augmentation(window)
        return torch.from_numpy(window).unsqueeze(-1)


def _extract_numeric_array(obj) -> np.ndarray:
    if obj is None:
        return np.array([], dtype=np.float32)
    if isinstance(obj, np.ndarray):
        arr = obj
    elif hasattr(obj, "to_numpy"):
        arr = obj.to_numpy()
    elif isinstance(obj, (list, tuple)):
        arr = np.asarray(obj)
    else:
        arr = np.asarray(obj)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = arr.astype(np.float32, copy=False)
    return arr[np.isfinite(arr)]


def _fast_parse_tsf(path: Path) -> List[np.ndarray]:
    """Custom fast parser for Monash .tsf files, ~100x faster than sktime."""
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        in_data = False
        for line in f:
            if in_data:
                # Format: series_name:start_timestamp:val1,val2,val3...
                parts = line.strip().split(':')
                if not parts:
                    continue
                vals_str = parts[-1]
                if not vals_str:
                    continue
                # Replace '?' with 'NaN' so numpy can parse it quickly
                vals_str = vals_str.replace('?', 'NaN')
                arr = np.fromstring(vals_str, dtype=np.float32, sep=',')
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    out.append(arr)
            elif line.startswith('@data'):
                in_data = True
    return out


def _fast_parse_ts(path: Path) -> List[np.ndarray]:
    """Custom fast parser for .ts files."""
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        in_data = False
        for line in f:
            if in_data:
                line = line.strip()
                if not line:
                    continue
                line = line.replace('?', 'NaN')
                # Multivariate uses ':' between dimensions
                line = line.replace(':', ',')
                arr = np.fromstring(line, dtype=np.float32, sep=',')
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    out.append(arr)
            elif line.startswith('@data'):
                in_data = True
    return out


def _safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return default


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _synthesize_from_feature_row(features: dict, length: int, rng: np.random.Generator) -> np.ndarray:
    length = int(max(16, length))
    mean = _safe_float(features.get("mean", 0.0), 0.0)
    var = max(_safe_float(features.get("var", 1.0), 1.0), 1e-6)
    std = float(np.sqrt(min(var, 1e6)))

    ar = _clip(_safe_float(features.get("x_acf1", 0.3), 0.3), -0.98, 0.98)
    trend_raw = _safe_float(features.get("trend", 0.0), 0.0)
    trend_slope = _clip(trend_raw, -5.0, 5.0) * (std / max(length, 1)) * 0.5

    seasonal_period = int(_safe_float(features.get("seasonal_period", 0.0), 0.0))
    has_seasonality = seasonal_period > 1 and seasonal_period <= (length // 2)
    seasonal_amp = std * (0.25 if has_seasonality else 0.0)

    spike_raw = _safe_float(features.get("spike", 0.0), 0.0)
    spike_prob = _clip(abs(spike_raw) * 0.02, 0.0, 0.15)
    spike_scale = std * (1.0 + _clip(abs(spike_raw), 0.0, 10.0))

    t = np.arange(length, dtype=np.float32)
    season = (
        seasonal_amp * np.sin((2.0 * np.pi * t) / max(seasonal_period, 1))
        if has_seasonality
        else np.zeros(length, dtype=np.float32)
    )
    base_noise = rng.normal(0.0, std * 0.35, size=length).astype(np.float32)
    random_walk = np.cumsum(rng.normal(0.0, std * 0.15, size=length).astype(np.float32))

    spike_mask = rng.random(size=length) < spike_prob
    spike_noise = rng.normal(0.0, spike_scale, size=length).astype(np.float32)
    spikes = np.where(spike_mask, spike_noise, 0.0).astype(np.float32)

    y = mean + (trend_slope * t) + season + base_noise + (ar * random_walk) + spikes
    y = y.astype(np.float32, copy=False)
    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std > 1e-6:
        y = ((y - y_mean) / y_std) * std + mean
    return y.astype(np.float32, copy=False)


def _load_feature_csv_as_synth_series(
    path: Path,
    synthetic_length: int,
    seed: int,
    progress_every_rows: int,
    use_polars: bool,
    max_rows_per_feature_file: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    try:
        if use_polars and HAS_POLARS:
            df_pl = pl.read_csv(
                str(path),
                infer_schema_length=1000,
                null_values=["NA", "NaN", "nan", ""],
                ignore_errors=True,
            )
            numeric_cols = [col for col, dtype in zip(df_pl.columns, df_pl.dtypes) if dtype.is_numeric()]
            if not numeric_cols:
                return []
            rows = df_pl.select(numeric_cols).to_dicts()
        else:
            df_pd = pd.read_csv(path)
            if df_pd.empty:
                return []
            numeric_df = df_pd.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            if numeric_df.empty:
                return []
            rows = numeric_df.to_dict(orient="records")
    except Exception as exc:
        print(f"[WARN] Failed reading feature CSV {path}: {exc}")
        return []

    total_rows = len(rows)
    print(f"[INFO] Synthesizing from {path} | rows={total_rows}")
    if max_rows_per_feature_file > 0 and total_rows > max_rows_per_feature_file:
        keep_idx = rng.choice(total_rows, size=max_rows_per_feature_file, replace=False)
        keep_idx.sort()
        rows = [rows[int(i)] for i in keep_idx]
        print(
            f"[INFO] {path.name}: sampled {len(rows)}/{total_rows} rows "
            f"(max_rows_per_feature_file={max_rows_per_feature_file})"
        )

    out: List[np.ndarray] = []
    final_rows = len(rows)
    for idx, row in enumerate(rows, start=1):
        series = _synthesize_from_feature_row(row, synthetic_length, rng)
        out.append(series)
        if progress_every_rows > 0 and idx % progress_every_rows == 0:
            print(f"[INFO] {path.name}: synthesized {idx}/{final_rows} rows")
    print(f"[INFO] Completed {path} | synthesized_series={len(out)}")
    return out


def load_all_series(
    data_dir: Path,
    *,
    feature_fallback: bool,
    synthetic_length: int,
    seed: int,
    progress_every_rows: int,
    feature_workers: int,
    max_rows_per_feature_file: int,
    exclude_dir: Path = None,
) -> Tuple[List[np.ndarray], List[Path], str]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    tsf_files = sorted(data_dir.rglob("*.tsf"))
    ts_files = sorted(data_dir.rglob("*.ts"))
    files = tsf_files + ts_files
    
    if exclude_dir is not None and exclude_dir.exists():
        exclude_path = str(exclude_dir.resolve())
        files = [p for p in files if not str(p.resolve()).startswith(exclude_path)]

    print(f"[INFO] Found raw files: tsf={len(tsf_files)} ts={len(ts_files)}")

    all_series: List[np.ndarray] = []
    loaded_files: List[Path] = []

    for file_idx, path in enumerate(files, start=1):
        print(f"[INFO] Loading raw file {file_idx}/{len(files)}: {path}")
        try:
            if path.suffix.lower() == ".tsf":
                series = _fast_parse_tsf(path)
            else:
                series = _fast_parse_ts(path)
            if series:
                all_series.extend(series)
                loaded_files.append(path)
                print(f"[INFO] Loaded {path.name}: series={len(series)} | total_series={len(all_series)}")
        except Exception as exc:
            print(f"[WARN] Failed loading {path}: {exc}")

    if all_series:
        return all_series, loaded_files, "raw"

    if files:
        raise RuntimeError("Found raw files but could not load any usable numeric series with sktime.")
    if not feature_fallback:
        csv_count = sum(1 for _ in data_dir.rglob("*.csv"))
        raise FileNotFoundError(
            f"No .tsf/.ts files found under {data_dir}. Found {csv_count} CSV files. "
            "Enable fallback with --feature-fallback to synthesize series from feature CSVs."
        )

    feature_csv_files = sorted([p for p in data_dir.rglob("*_features.csv") if p.is_file()])
    print(f"[INFO] Feature fallback enabled. Found feature CSV files: {len(feature_csv_files)}")
    backend = "polars" if HAS_POLARS else "pandas"
    print(f"[INFO] Feature CSV backend: {backend}")
    if not feature_csv_files:
        csv_count = sum(1 for _ in data_dir.rglob("*.csv"))
        raise FileNotFoundError(
            f"No .tsf/.ts files and no *_features.csv files found under {data_dir}. "
            f"CSV files present: {csv_count}."
        )

    synthetic_series: List[np.ndarray] = []
    used_files: List[Path] = []
    workers = max(1, feature_workers)
    print(f"[INFO] Feature fallback workers: {workers}")
    if workers == 1:
        for file_idx, path in enumerate(feature_csv_files, start=1):
            print(f"[INFO] Processing feature file {file_idx}/{len(feature_csv_files)}: {path}")
            rows = _load_feature_csv_as_synth_series(
                path,
                synthetic_length=synthetic_length,
                seed=seed + file_idx,
                progress_every_rows=progress_every_rows,
                use_polars=True,
                max_rows_per_feature_file=max_rows_per_feature_file,
            )
            if rows:
                synthetic_series.extend(rows)
                used_files.append(path)
                print(
                    f"[INFO] Added {len(rows)} synthetic series from {path.name} | "
                    f"total_series={len(synthetic_series)}"
                )
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for file_idx, path in enumerate(feature_csv_files, start=1):
                print(f"[INFO] Queued feature file {file_idx}/{len(feature_csv_files)}: {path}")
                fut = ex.submit(
                    _load_feature_csv_as_synth_series,
                    path,
                    synthetic_length,
                    seed + file_idx,
                    progress_every_rows,
                    True,
                    max_rows_per_feature_file,
                )
                futures[fut] = path

            for fut in as_completed(futures):
                path = futures[fut]
                try:
                    rows = fut.result()
                    if rows:
                        synthetic_series.extend(rows)
                        used_files.append(path)
                        print(
                            f"[INFO] Added {len(rows)} synthetic series from {path.name} | "
                            f"total_series={len(synthetic_series)}"
                        )
                except Exception as exc:
                    print(f"[WARN] Failed processing {path}: {exc}")

    if not synthetic_series:
        raise RuntimeError("Feature fallback was enabled, but no usable numeric feature rows were found.")

    return synthetic_series, used_files, "feature-fallback"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _write_metrics(metrics_out: Path, payload: dict) -> None:
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class EMA:
    """Exponential Moving Average of model parameters for smoother convergence."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def apply_to(self, model: nn.Module) -> None:
        """Replace model weights with EMA shadow weights."""
        model.load_state_dict(self.shadow, strict=False)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a simple TSFM with masked patch reconstruction.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--real-data-dir",
        type=Path,
        default=None,
        help="Directory with real .tsf/.ts pretraining corpora. Loaded alongside synthetic data.",
    )
    parser.add_argument(
        "--real-data-only",
        action="store_true",
        default=False,
        help="Skip synthetic feature fallback; use only real corpora from --real-data-dir.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers. -1 = auto (all CPU cores).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument(
        "--feature-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If no raw .tsf/.ts files are found, synthesize series from *_features.csv files.",
    )
    parser.add_argument(
        "--synthetic-length",
        type=int,
        default=1024,
        help="Length of synthesized series when feature fallback is used.",
    )
    parser.add_argument(
        "--progress-every-rows",
        type=int,
        default=5000,
        help="When using feature fallback, print progress every N rows per CSV file. 0 disables row-level progress.",
    )
    parser.add_argument(
        "--feature-workers",
        type=int,
        default=-1,
        help="Worker threads for feature-fallback CSV processing. -1 = auto (all CPU cores).",
    )
    parser.add_argument(
        "--max-rows-per-feature-file",
        type=int,
        default=0,
        help="Maximum rows to synthesize per *_features.csv file. 0 means use all rows.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps for larger effective batch size.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        default=False,
        help="Disable automatic mixed precision training.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        default=False,
        help="Disable torch.compile (PyTorch 2.0+) model optimization.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop if val loss does not improve for N epochs (0 = disabled).",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-5,
        help="Minimum improvement in val loss to reset patience counter.",
    )
    parser.add_argument(
        "--best-model-path",
        type=Path,
        default=Path("tsfm_best.pt"),
        help="Path to save the best model checkpoint (by validation loss).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Name for this run (used in metrics output).",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Path to write per-epoch metrics JSON.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Override epoch number when resuming.",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        choices=["mse", "huber"],
        default="huber",
        help="Loss function for pretraining (huber is more robust to outliers).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Use gradient checkpointing to trade compute for memory (saves ~40%% VRAM).",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay rate for weight averaging (0 = disabled).",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable time-series data augmentation during training.",
    )
    return parser.parse_args()

def validate(model, val_loader, device, mask_ratio=0.5, criterion=None):
    """Validate the model using the same mask_ratio and loss as training."""
    if criterion is None:
        criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            reconstructed, original, mask = model(batch, mask_ratio=mask_ratio)

            if not mask.any():
                continue

            preds = reconstructed[mask]
            targets = original[mask]

            loss = criterion(preds, targets)

            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    set_seed(args.seed)
    started_at = datetime.now().isoformat(timespec="seconds")

    if not (0.0 < args.mask_ratio < 1.0):
        raise ValueError("mask_ratio must be between 0 and 1")

    # --- Auto-detect optimal resource counts ---
    cpu_count = os.cpu_count() or 4
    if args.num_workers < 0:
        args.num_workers = cpu_count
    if args.feature_workers < 0:
        args.feature_workers = cpu_count

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    # --- CUDA optimizations ---
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print(f"Device: {device}")
    print(f"CPU cores: {cpu_count} | DataLoader workers: {args.num_workers} | Feature workers: {args.feature_workers}")
    print(f"Mixed precision (AMP): {use_amp}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Loading datasets from: {args.data_dir}")

    all_series: List[np.ndarray] = []
    all_loaded_files: List[Path] = []
    source_modes: List[str] = []
    synthetic_series_count = 0
    real_series_count = 0

    if args.real_data_only:
        args.feature_fallback = False

    if not args.real_data_only:
        series_list, loaded_files, source_mode = load_all_series(
            args.data_dir,
            feature_fallback=args.feature_fallback,
            synthetic_length=args.synthetic_length,
            seed=args.seed,
            progress_every_rows=args.progress_every_rows,
            feature_workers=args.feature_workers,
            max_rows_per_feature_file=args.max_rows_per_feature_file,
            exclude_dir=args.real_data_dir,
        )
        all_series.extend(series_list)
        all_loaded_files.extend(loaded_files)
        source_modes.append(source_mode)
        synthetic_series_count += len(series_list)

    if args.real_data_dir is not None and args.real_data_dir.exists():
        print(f"[INFO] Loading real corpora from: {args.real_data_dir}")
        real_series, real_files, real_mode = load_all_series(
            args.real_data_dir,
            feature_fallback=False,
            synthetic_length=args.synthetic_length,
            seed=args.seed,
            progress_every_rows=args.progress_every_rows,
            feature_workers=args.feature_workers,
            max_rows_per_feature_file=0,
            exclude_dir=None,
        )
        all_series.extend(real_series)
        all_loaded_files.extend(real_files)
        source_modes.append(real_mode)
        real_series_count += len(real_series)
        print(f"[INFO] Real corpora: {len(real_series)} series from {len(real_files)} files")

    if not all_series:
        raise RuntimeError("No series loaded. Check --data-dir and --real-data-dir paths.")

    series_list = all_series
    loaded_files = all_loaded_files

    if "raw" in source_modes and "feature-fallback" in source_modes:
        source_mode = "mixed"
    elif "raw" in source_modes:
        source_mode = "raw"
    elif "feature-fallback" in source_modes:
        source_mode = "feature-fallback"
    else:
        source_mode = source_modes[0]
    print(f"Data source mode: {source_mode}")
    print(f"Loaded files: {len(loaded_files)}")
    print(f"Loaded series: {len(series_list)}")

    print("[INFO] Building sliding-window datasets...")
    
    from torch.utils.data import random_split

    # Create full dataset WITHOUT augmentation for splitting
    full_dataset = WindowDataset(
        series_list,
        context_length=args.context_length,
        stride=args.stride,
        augment=False,  # No augmentation on base dataset
    )

    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size]
    )

    # Create separate datasets: train WITH augmentation, val WITHOUT
    train_dataset = WindowDataset(
        series_list,
        context_length=args.context_length,
        stride=args.stride,
        augment=args.augment,
    )
    # Use the same base dataset for validation (augment=False)
    dataset = full_dataset  # for len() reporting below
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices.indices)

    print("[INFO] DataLoader initialization...")
    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4
    
    train_loader = DataLoader(train_dataset, **dl_kwargs)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TSFMPretrain(
        context_length=args.context_length,
        patch_length=args.patch_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    # --- torch.compile for graph optimization (PyTorch 2.0+) ---
    if not args.no_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile enabled")
        except Exception as exc:
            print(f"[WARN] torch.compile failed, continuing without: {exc}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=use_amp)

    # --- Loss function selection ---
    if args.loss_fn == "huber":
        criterion = nn.SmoothL1Loss(beta=1.0)
        print("[INFO] Loss function: Huber (SmoothL1Loss, beta=1.0)")
    else:
        criterion = nn.MSELoss()
        print("[INFO] Loss function: MSE")

    steps_per_epoch = args.max_steps_per_epoch if args.max_steps_per_epoch > 0 else len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # --- EMA initialization ---
    ema = None
    if args.ema_decay > 0:
        ema = EMA(model, decay=args.ema_decay)
        print(f"[INFO] EMA enabled with decay={args.ema_decay}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Total windows: {len(dataset)}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Augmentation: {args.augment}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print("Starting masked pretraining...")

    run_name = args.run_name.strip() or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    real_ratio = real_series_count / max(real_series_count + synthetic_series_count, 1) if (real_series_count + synthetic_series_count) > 0 else 0.0
    metrics_payload = {
        "run_name": run_name,
        "started_at": started_at,
        "finished_at": None,
        "config": _as_jsonable(vars(args)),
        "system": {
            "device": str(device),
            "cpu_count": cpu_count,
            "use_amp": bool(use_amp),
        },
        "data": {
            "source_mode": source_mode,
            "loaded_files": len(loaded_files),
            "loaded_series": len(series_list),
            "total_windows": len(dataset),
            "composition": {
                "synthetic_series": int(synthetic_series_count),
                "real_series": int(real_series_count),
                "real_ratio": float(real_ratio),
            },
        },
        "model": {
            "total_params": int(total_params),
        },
        "epochs": [],
        "summary": {},
    }
    if args.metrics_out is not None:
        _write_metrics(args.metrics_out, metrics_payload)

    accum_steps = max(1, args.gradient_accumulation_steps)

    total_batches = len(train_loader)
    
    start_epoch = 1
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[INFO] Loading checkpoint from {args.resume_from}...")
        # weights_only=False is needed to load optimizer/scheduler state dicts
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        
        # Load states
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        
        # NOTE: If changing batch size drastically, the old scheduler step might be mismatched.
        # We load it, but keep an eye on your learning rate prints.
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        
        # Determine the correct start epoch
        if args.start_epoch is not None:
            start_epoch = args.start_epoch
        elif "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        else:
            # Fallback for your old checkpoints
            start_epoch = 12 
            
        print(f"[INFO] Resuming training from Epoch {start_epoch}")

    # Create a directory for checkpoints
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = None

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start_perf = time.perf_counter()
        running_loss = 0.0
        step_count = 0
        micro_step = 0

        max_steps = total_batches
        if args.max_steps_per_epoch > 0:
            max_steps = min(args.max_steps_per_epoch, total_batches)

        pbar = tqdm(
            train_loader,
            total=max_steps,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            colour="green",
        )

        for batch in pbar:
            if args.max_steps_per_epoch > 0 and step_count >= args.max_steps_per_epoch:
                break

            batch = batch.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                reconstructed, original, mask = model(batch, mask_ratio=args.mask_ratio)
                if mask.sum() == 0:
                    continue

                preds = reconstructed[mask]
                targets = original[mask].detach()
                loss = criterion(preds, targets) / accum_steps

            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema is not None:
                    ema.update(model)

            running_loss += loss.item() * accum_steps
            step_count += 1

            # --- Update progress bar with live stats ---
            avg_loss = running_loss / step_count
            postfix = {"loss": f"{avg_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
            if device.type == "cuda":
                gpu_mb = torch.cuda.max_memory_allocated() / 1024**2
                postfix["gpu_peak"] = f"{gpu_mb:.0f}MB"
            pbar.set_postfix(postfix)


            global_step = (epoch - 1) * max_steps + step_count

            if global_step % args.save_every == 0:
                # Saves inside the 'checkpoints' folder and includes the epoch number
                ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{global_step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                    "epoch": epoch, 
                }, ckpt_path)

        pbar.close()

        # Flush remaining accumulated gradients
        if micro_step % accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = running_loss / max(step_count, 1)
        epoch_time_sec = time.perf_counter() - epoch_start_perf

        # Track GPU peak memory per epoch
        gpu_peak_mb = None
        if device.type == "cuda":
            gpu_peak_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
            torch.cuda.reset_peak_memory_stats()

        epoch_metrics = {
            "epoch": epoch,
            "steps": int(step_count),
            "masked_mse": float(avg_loss),
            "epoch_time_sec": float(epoch_time_sec),
        }
        if gpu_peak_mb is not None:
            epoch_metrics["gpu_peak_mb"] = gpu_peak_mb

        metrics_payload["epochs"].append(epoch_metrics)
        if args.metrics_out is not None:
            _write_metrics(args.metrics_out, metrics_payload)
        print(f"\nEpoch {epoch}/{args.epochs} | Steps {step_count} | Train Loss {avg_loss:.6f}")

        # --- Validation with correct mask_ratio and loss fn ---
        val_loss = validate(model, val_loader, device, mask_ratio=args.mask_ratio, criterion=criterion)
        print(f"Validation Loss: {val_loss:.6f}")

        # Also validate EMA model if enabled
        ema_val_loss = None
        if ema is not None:
            original_state = copy.deepcopy(model.state_dict())
            ema.apply_to(model)
            ema_val_loss = validate(model, val_loader, device, mask_ratio=args.mask_ratio, criterion=criterion)
            model.load_state_dict(original_state)
            print(f"EMA Validation Loss: {ema_val_loss:.6f}")
            epoch_metrics["ema_val_loss"] = float(ema_val_loss)

        # Use whichever is better: regular or EMA
        effective_val_loss = val_loss
        if ema_val_loss is not None and ema_val_loss < val_loss:
            effective_val_loss = ema_val_loss

        epoch_metrics["val_loss"] = float(val_loss)

        if effective_val_loss < best_val_loss - args.early_stopping_min_delta:
            best_val_loss = effective_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save the better state (EMA or regular)
            if ema is not None and ema_val_loss is not None and ema_val_loss < val_loss:
                best_state_dict = ema.state_dict()
                best_state_dict = {k: v.cpu() for k, v in best_state_dict.items()}
            else:
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if args.best_model_path is not None:
                torch.save({
                    "model_state_dict": best_state_dict,
                    "epoch": epoch,
                    "val_loss": effective_val_loss,
                    "args": vars(args),
                }, args.best_model_path)
                print(f"[INFO] New best model saved to {args.best_model_path} (val_loss={effective_val_loss:.6f})")
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch} (best val_loss={best_val_loss:.6f} at epoch {best_epoch})")
            break

    ckpt_path = Path("tsfm_pretrain.pt")
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path.resolve()}")

    if args.metrics_out is not None:
        metrics_payload["summary"] = {
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "final_val_loss": float(val_loss),
            "total_epochs": int(epoch),
            "early_stopped": bool(epochs_without_improvement >= args.early_stopping_patience > 0),
            "data_composition": {
                "synthetic_series": int(synthetic_series_count),
                "real_series": int(real_series_count),
                "real_ratio": float(real_ratio),
            },
        }
        _write_metrics(args.metrics_out, metrics_payload)


if __name__ == "__main__":
    main()
