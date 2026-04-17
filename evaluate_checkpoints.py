"""
Benchmark evaluator for TSFM checkpoints and TimesFM baseline.

This script adds a unified evaluation path over TimesFM paper-style datasets:
  - ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Weather

SCALING / OUTPUT RECONCILIATION RULES
-------------------------------------
Rule 1 (TimesFM):
  Use TimesFM's own normalization path.
  - v2.5 API: ForecastConfig(normalize_inputs=True)
  - v1 API: forecast(..., normalize=True)
  Do NOT manually inverse-scale TimesFM outputs afterwards.

Rule 2 (TSFM):
  To stay consistent with this repository's fine-tuning setup, TSFM inputs are
  z-normalized by train-split statistics by default, and TSFM outputs are then
  inverse-scaled back to raw units before metric computation.

Rule 3 (Metrics):
  MSE/MAE are always computed in raw dataset units against raw targets.

Positive examples:
  - Correct: TimesFM normalize_inputs=True, evaluate point_forecast directly
    against raw y_true.
  - Correct: TSFM receives z-scored context, then prediction is inverse-scaled
    with the SAME train mean/std before comparing to raw y_true.

Negative examples:
  - Wrong: TimesFM normalize_inputs=True AND then multiplying forecasts by
    train std + mean again (double inverse-scaling).
  - Wrong: Manually z-scoring TimesFM inputs while also normalize_inputs=True
    (double normalization).
  - Wrong: Comparing TSFM normalized predictions directly to raw y_true.
"""

import argparse
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from finetune_forecasting import TSFMForForecasting


_EPS = 1e-6


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    target_candidates: Tuple[str, ...]
    freq_bucket: int
    split_policy: str


@dataclass
class DatasetPayload:
    name: str
    csv_path: Path
    target_column: str
    freq_bucket: int
    split_info: Dict[str, Any]
    train_mean: float
    train_std: float
    contexts: np.ndarray
    targets: np.ndarray


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "ETTh1": DatasetSpec("ETTh1", "ETTh1.csv", ("OT",), 0, "ett_hourly"),
    "ETTh2": DatasetSpec("ETTh2", "ETTh2.csv", ("OT",), 0, "ett_hourly"),
    "ETTm1": DatasetSpec("ETTm1", "ETTm1.csv", ("OT",), 0, "ett_minutely"),
    "ETTm2": DatasetSpec("ETTm2", "ETTm2.csv", ("OT",), 0, "ett_minutely"),
    "Electricity": DatasetSpec("Electricity", "electricity.csv", ("OT",), 0, "ratio"),
    "Traffic": DatasetSpec("Traffic", "traffic.csv", ("OT", "traffic_volume", "Traffic_Volume"), 0, "ratio"),
    "Weather": DatasetSpec("Weather", "weather.csv", ("OT",), 0, "ratio"),
}


class ContextDataset(Dataset):
    """Torch dataset for batched TSFM inference on context windows."""

    def __init__(self, contexts: np.ndarray):
        self.contexts = contexts.astype(np.float32)

    def __len__(self) -> int:
        return self.contexts.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.contexts[idx]).unsqueeze(-1)


def _find_dataset_csv(data_dir: Path, filename: str) -> Optional[Path]:
    """Finds a dataset csv by exact filename (case-insensitive), recursively."""
    direct = data_dir / filename
    if direct.exists():
        return direct

    candidates = [
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.name.lower() == filename.lower()
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return candidates[0]


def _pick_target_column(df: pd.DataFrame, target_candidates: Sequence[str]) -> str:
    """Resolves the target column from candidates, then falls back to numeric columns."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in target_candidates:
        col = lower_map.get(cand.lower())
        if col is not None:
            return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns available for univariate evaluation.")

    # Use first numeric feature by default for electricity/traffic style datasets.
    return numeric_cols[0]


def _load_univariate_series(spec: DatasetSpec, data_dir: Path) -> Tuple[np.ndarray, Path, str]:
    """Loads and cleans a univariate series for the requested dataset spec."""
    csv_path = _find_dataset_csv(data_dir, spec.filename)
    if csv_path is None and spec.name == "Traffic":
        # Allow direct use of Metro Interstate Traffic Volume CSV as Traffic benchmark input.
        metro_candidates = [
            data_dir / "metro+interstate+traffic+volume" / "Metro_Interstate_Traffic_Volume.csv",
            data_dir / "metro_volume.csv",
        ]
        for candidate in metro_candidates:
            if candidate.exists():
                csv_path = candidate
                break

    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find {spec.filename} under {data_dir}. "
            "Run prepare_datasets.py or place the csv manually."
        )

    df = pd.read_csv(csv_path)
    target_col = _pick_target_column(df, spec.target_candidates)
    values = pd.to_numeric(df[target_col], errors="coerce")
    values = values.ffill().bfill()

    if values.isna().any():
        values = values.dropna()

    arr = values.to_numpy(dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Loaded invalid series from {csv_path} using column {target_col}.")

    return arr, csv_path, target_col


def _resolve_splits(spec: DatasetSpec, n: int) -> Dict[str, Any]:
    """Resolves train/val/test boundaries using canonical LTSF splits where applicable."""
    if spec.split_policy == "ett_hourly":
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_start = val_end
        policy = "ett_hourly_12m_4m_4m"
    elif spec.split_policy == "ett_minutely":
        mul = 4
        train_end = 12 * 30 * 24 * mul
        val_end = train_end + 4 * 30 * 24 * mul
        test_start = val_end
        policy = "ett_minutely_12m_4m_4m"
    else:
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        test_start = val_end
        policy = "ratio_70_10_20"

    # Fall back to ratio split if fixed boundaries do not fit available data.
    if test_start >= n:
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        test_start = val_end
        policy = "ratio_70_10_20_fallback"

    return {
        "policy": policy,
        "train_end": int(train_end),
        "val_end": int(val_end),
        "test_start": int(test_start),
        "n_total": int(n),
    }


def _safe_std(x: np.ndarray) -> float:
    std = float(np.std(x))
    return std if std > _EPS else 1.0


def _build_test_windows(
    series: np.ndarray,
    test_start: int,
    context_length: int,
    horizon: int,
    stride: int,
    max_windows: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Builds rolling test windows with targets entirely in the test segment."""
    if stride <= 0:
        raise ValueError("stride must be > 0")

    contexts: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    end = len(series) - horizon
    for anchor in range(test_start, end + 1, stride):
        c_start = anchor - context_length
        if c_start < 0:
            continue

        context = series[c_start:anchor]
        target = series[anchor : anchor + horizon]
        if len(context) != context_length or len(target) != horizon:
            continue

        contexts.append(context)
        targets.append(target)

        if max_windows is not None and len(contexts) >= max_windows:
            break

    if not contexts:
        raise ValueError(
            "No valid evaluation windows were produced. "
            "Try reducing context_length / horizon or using a smaller stride."
        )

    return np.stack(contexts).astype(np.float32), np.stack(targets).astype(np.float32)


def _extract_state_dict(checkpoint_obj: Any) -> Dict[str, torch.Tensor]:
    """Extracts a tensor state-dict from common checkpoint container formats."""
    obj = checkpoint_obj
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise ValueError("Checkpoint does not contain a recognizable state_dict.")

    cleaned: Dict[str, torch.Tensor] = {}
    for key, val in obj.items():
        if not torch.is_tensor(val):
            continue
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[7:]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        cleaned[new_key] = val

    if not cleaned:
        raise ValueError("No tensor weights found in checkpoint.")

    return cleaned


def _compute_load_quality(
    model: TSFMForForecasting,
    missing: Sequence[str],
    unexpected: Sequence[str],
) -> Dict[str, Any]:
    """Summarizes checkpoint load quality for sanity checks and reporting."""
    total_model_keys = len(model.state_dict())
    missing_count = len(missing)
    loaded_count = max(0, total_model_keys - missing_count)
    load_ratio = (loaded_count / total_model_keys) if total_model_keys > 0 else 0.0
    return {
        "total_model_keys": int(total_model_keys),
        "loaded_keys": int(loaded_count),
        "missing_keys_count": int(missing_count),
        "unexpected_keys_count": int(len(unexpected)),
        "load_ratio": float(load_ratio),
    }


def _infer_architecture_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Infers TSFM architecture params from checkpoint tensor shapes when possible."""
    inferred: Dict[str, int] = {}

    pos_emb = state_dict.get("patch_embedding.position_embedding")
    if torch.is_tensor(pos_emb) and pos_emb.ndim == 3:
        inferred["embed_dim"] = int(pos_emb.shape[-1])

    val_emb_w = state_dict.get("patch_embedding.value_embedding.weight")
    if torch.is_tensor(val_emb_w) and val_emb_w.ndim == 3:
        inferred.setdefault("embed_dim", int(val_emb_w.shape[0]))
        inferred["patch_length"] = int(val_emb_w.shape[-1])

    layer_regex = re.compile(r"^transformer\.encoder\.layers\.(\d+)\.")
    layer_ids: List[int] = []
    for key in state_dict.keys():
        m = layer_regex.match(key)
        if m:
            layer_ids.append(int(m.group(1)))
    if layer_ids:
        inferred["num_layers"] = max(layer_ids) + 1

    head_weight = state_dict.get("forecasting_head.projection.weight")
    head_bias = state_dict.get("forecasting_head.projection.bias")
    if torch.is_tensor(head_weight) and head_weight.ndim == 2:
        inferred["model_forecast_horizon"] = int(head_weight.shape[0])
    elif torch.is_tensor(head_bias) and head_bias.ndim == 1:
        inferred["model_forecast_horizon"] = int(head_bias.shape[0])

    return inferred


def _torch_load_checkpoint(path: Path, device: torch.device) -> Any:
    """Loads checkpoints across common PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_tsfm_model(
    checkpoint_path: Path,
    device: torch.device,
    context_length: int,
    patch_length: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    forecast_horizon: int,
    pooling: str,
) -> Tuple[TSFMForForecasting, Dict[str, Any]]:
    """Loads a TSFM forecasting model checkpoint."""
    ckpt_obj = _torch_load_checkpoint(checkpoint_path, device)
    state_dict = _extract_state_dict(ckpt_obj)

    inferred_arch = _infer_architecture_from_state_dict(state_dict)
    patch_length = int(inferred_arch.get("patch_length", patch_length))
    embed_dim = int(inferred_arch.get("embed_dim", embed_dim))
    num_layers = int(inferred_arch.get("num_layers", num_layers))
    model_forecast_horizon = int(
        inferred_arch.get("model_forecast_horizon", forecast_horizon)
    )

    model = TSFMForForecasting(
        context_length=context_length,
        patch_length=patch_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        forecast_horizon=model_forecast_horizon,
        pooling=pooling,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    quality = _compute_load_quality(model, missing, unexpected)

    return model, {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "quality": quality,
        "inferred_architecture": {
            "patch_length": patch_length,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "model_forecast_horizon": model_forecast_horizon,
        },
    }


@torch.no_grad()
def _predict_tsfm(
    model: TSFMForForecasting,
    contexts: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Runs batched TSFM inference and returns [N, H] predictions."""
    loader = DataLoader(
        ContextDataset(contexts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    preds: List[np.ndarray] = []
    for batch_x in loader:
        batch_x = batch_x.to(device)
        batch_pred = model(batch_x).squeeze(-1).detach().cpu().numpy()
        preds.append(batch_pred)

    return np.concatenate(preds, axis=0).astype(np.float32)


@torch.no_grad()
def _predict_tsfm_with_rollout(
    model: TSFMForForecasting,
    contexts: np.ndarray,
    target_horizon: int,
    model_horizon: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predicts target_horizon steps, using iterative rollout when needed."""
    if target_horizon <= model_horizon:
        pred = _predict_tsfm(model=model, contexts=contexts, batch_size=batch_size, device=device)
        return pred[:, :target_horizon]

    ctx = contexts.astype(np.float32).copy()
    chunks: List[np.ndarray] = []
    remaining = int(target_horizon)

    while remaining > 0:
        step_pred = _predict_tsfm(model=model, contexts=ctx, batch_size=batch_size, device=device)
        take = min(model_horizon, remaining)
        chunk = step_pred[:, :take].astype(np.float32)
        chunks.append(chunk)
        remaining -= take
        if remaining > 0:
            ctx = np.concatenate([ctx[:, take:], chunk], axis=1)

    return np.concatenate(chunks, axis=1).astype(np.float32)


def _reconcile_point_forecast(
    point_forecast: Any,
    expected_batch: int,
    horizon: int,
    source: str,
) -> np.ndarray:
    """
    Reconciles forecast outputs to strict [batch, horizon] point predictions.

    Positive example:
      [batch, horizon] -> accepted directly.

    Negative example:
      [batch, horizon, quantiles] -> rejected here (ambiguous for point path).
    """
    arr = np.asarray(point_forecast)

    if arr.ndim == 1:
        if expected_batch != 1:
            raise ValueError(f"{source} returned 1D output for batch={expected_batch}.")
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(
            f"{source} returned unexpected point forecast shape {arr.shape}. "
            "Expected [batch, horizon]."
        )

    if arr.shape[0] != expected_batch:
        if arr.shape[1] == expected_batch:
            arr = arr.T
        else:
            raise ValueError(
                f"{source} batch mismatch: got {arr.shape[0]}, expected {expected_batch}."
            )

    if arr.shape[1] < horizon:
        raise ValueError(
            f"{source} horizon mismatch: got {arr.shape[1]} < requested {horizon}."
        )

    if arr.shape[1] > horizon:
        arr = arr[:, :horizon]

    return arr.astype(np.float32)


class TimesFMBaseline:
    """Compatibility wrapper for TimesFM v2.5 and legacy v1 API surfaces."""

    def __init__(
        self,
        version: str,
        repo_id: Optional[str],
        horizon: int,
        max_context: int,
        batch_size: int,
        normalize_inputs: bool,
        force_cpu: bool,
    ):
        try:
            import timesfm  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "timesfm is not installed. Install with: pip install timesfm[torch]"
            ) from exc

        self.timesfm = timesfm
        self.horizon = horizon
        self.max_context = max_context
        self.batch_size = max(1, int(batch_size))
        self.normalize_inputs = bool(normalize_inputs)
        self.force_cpu = bool(force_cpu)

        self.impl = ""
        self.model = None
        self._forecast_sig = None

        self._init_model(version=version, repo_id=repo_id)

    def _init_model(self, version: str, repo_id: Optional[str]) -> None:
        # Prefer TimesFM 2.5 API when available.
        if version in ("auto", "2p5") and hasattr(self.timesfm, "TimesFM_2p5_200M_torch"):
            self.impl = "timesfm_2p5"
            repo = repo_id or "google/timesfm-2.5-200m-pytorch"
            self.model = self.timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo)

            fc_kwargs = {
                "max_context": self.max_context,
                "max_horizon": self.horizon,
                "normalize_inputs": self.normalize_inputs,
                "per_core_batch_size": self.batch_size,
                "use_continuous_quantile_head": True,
                "force_flip_invariance": True,
                "infer_is_positive": True,
                "fix_quantile_crossing": True,
            }

            fc_sig = inspect.signature(self.timesfm.ForecastConfig)
            filtered_fc_kwargs = {
                k: v for k, v in fc_kwargs.items() if k in fc_sig.parameters
            }

            self.model.compile(self.timesfm.ForecastConfig(**filtered_fc_kwargs))
            self._forecast_sig = inspect.signature(self.model.forecast)
            return

        # Fall back to v1 API.
        if version in ("auto", "v1") and hasattr(self.timesfm, "TimesFm"):
            self.impl = "timesfm_v1"

            backend = "cpu"
            if not self.force_cpu and torch.cuda.is_available():
                backend = "gpu"

            hparams_kwargs = {}
            hparams_sig = inspect.signature(self.timesfm.TimesFmHparams)
            if "backend" in hparams_sig.parameters:
                hparams_kwargs["backend"] = backend
            if "per_core_batch_size" in hparams_sig.parameters:
                hparams_kwargs["per_core_batch_size"] = self.batch_size
            if "horizon_len" in hparams_sig.parameters:
                hparams_kwargs["horizon_len"] = self.horizon
            if "context_len" in hparams_sig.parameters:
                hparams_kwargs["context_len"] = self.max_context

            ckpt_kwargs = {}
            ckpt_sig = inspect.signature(self.timesfm.TimesFmCheckpoint)
            if "huggingface_repo_id" in ckpt_sig.parameters:
                ckpt_kwargs["huggingface_repo_id"] = (
                    repo_id or "google/timesfm-1.0-200m-pytorch"
                )

            self.model = self.timesfm.TimesFm(
                hparams=self.timesfm.TimesFmHparams(**hparams_kwargs),
                checkpoint=self.timesfm.TimesFmCheckpoint(**ckpt_kwargs),
            )
            self._forecast_sig = inspect.signature(self.model.forecast)
            return

        raise RuntimeError(
            "Unable to initialize TimesFM from installed package. "
            "Expected TimesFM_2p5_200M_torch or TimesFm API."
        )

    def predict(self, contexts: np.ndarray, freq_bucket: int) -> np.ndarray:
        """Runs TimesFM forecasts in batches and returns [N, H] point outputs."""
        if self.model is None:
            raise RuntimeError("TimesFM model is not initialized.")

        outputs: List[np.ndarray] = []
        for i in range(0, len(contexts), self.batch_size):
            batch_ctx = contexts[i : i + self.batch_size]
            inputs = [np.asarray(x, dtype=np.float32) for x in batch_ctx]

            if self.impl == "timesfm_2p5":
                point_forecast, _ = self.model.forecast(horizon=self.horizon, inputs=inputs)
            else:
                kwargs = {}
                if "inputs" in self._forecast_sig.parameters:
                    kwargs["inputs"] = inputs
                if "freq" in self._forecast_sig.parameters:
                    kwargs["freq"] = [freq_bucket] * len(inputs)
                if "normalize" in self._forecast_sig.parameters:
                    kwargs["normalize"] = self.normalize_inputs
                if "forecast_context_len" in self._forecast_sig.parameters:
                    kwargs["forecast_context_len"] = self.max_context

                point_forecast, _ = self.model.forecast(**kwargs)

            point = _reconcile_point_forecast(
                point_forecast,
                expected_batch=len(inputs),
                horizon=self.horizon,
                source=f"TimesFM({self.impl})",
            )
            outputs.append(point)

        return np.concatenate(outputs, axis=0).astype(np.float32)


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    err = pred - target
    mse = float(np.mean(np.square(err)))
    mae = float(np.mean(np.abs(err)))
    return {"mse": mse, "mae": mae}


def _resolve_checkpoints(paths: Sequence[str], glob_pattern: Optional[str]) -> List[Path]:
    resolved: List[Path] = []

    for p in paths:
        path = Path(p)
        if path.exists() and path.is_file():
            resolved.append(path)
        else:
            print(f"[WARN] Skipping missing checkpoint: {p}")

    if glob_pattern:
        for p in sorted(Path(".").glob(glob_pattern)):
            if p.is_file():
                resolved.append(p)

    # De-duplicate while preserving order.
    uniq: List[Path] = []
    seen = set()
    for p in resolved:
        rp = str(p.resolve())
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)

    return uniq


def _prepare_payloads(args: argparse.Namespace) -> List[DatasetPayload]:
    payloads: List[DatasetPayload] = []

    stride = args.stride if args.stride is not None else args.forecast_horizon

    for dataset_name in args.datasets:
        spec = DATASET_SPECS[dataset_name]

        try:
            series, csv_path, target_col = _load_univariate_series(spec, args.data_dir)
        except Exception as exc:
            if args.strict_datasets:
                raise
            print(f"[WARN] Skipping {dataset_name}: {exc}")
            continue

        split_info = _resolve_splits(spec, len(series))

        try:
            contexts, targets = _build_test_windows(
                series=series,
                test_start=split_info["test_start"],
                context_length=args.context_length,
                horizon=args.forecast_horizon,
                stride=stride,
                max_windows=args.max_test_windows,
            )
        except Exception as exc:
            if args.strict_datasets:
                raise
            print(f"[WARN] Skipping {dataset_name}: {exc}")
            continue

        train_slice = series[: split_info["train_end"]]
        if len(train_slice) == 0:
            train_slice = series

        train_mean = float(np.mean(train_slice))
        train_std = _safe_std(train_slice)

        payloads.append(
            DatasetPayload(
                name=dataset_name,
                csv_path=csv_path,
                target_column=target_col,
                freq_bucket=spec.freq_bucket,
                split_info=split_info,
                train_mean=train_mean,
                train_std=train_std,
                contexts=contexts,
                targets=targets,
            )
        )

    return payloads


def _aggregate_metrics(dataset_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, Dict[str, List[float]]] = {}
    rank_by_model: Dict[str, List[float]] = {}
    rel_mse_vs_timesfm: Dict[str, List[float]] = {}
    rel_mae_vs_timesfm: Dict[str, List[float]] = {}

    for result in dataset_results.values():
        models = result["models"]
        for model_key, metrics in models.items():
            by_model.setdefault(model_key, {"mse": [], "mae": []})
            by_model[model_key]["mse"].append(metrics["mse"])
            by_model[model_key]["mae"].append(metrics["mae"])

        # Scale-robust ranking per dataset (1 is best/lower error).
        ranked_mse = sorted(models.items(), key=lambda item: float(item[1]["mse"]))
        for rank_idx, (model_key, _) in enumerate(ranked_mse, start=1):
            rank_by_model.setdefault(model_key, []).append(float(rank_idx))

        # Relative metrics against TimesFM within each dataset (unitless).
        timesfm_metrics = models.get("timesfm")
        if timesfm_metrics is not None:
            timesfm_mse = max(float(timesfm_metrics["mse"]), _EPS)
            timesfm_mae = max(float(timesfm_metrics["mae"]), _EPS)
            for model_key, metrics in models.items():
                if model_key == "timesfm":
                    continue
                rel_mse_vs_timesfm.setdefault(model_key, []).append(float(metrics["mse"]) / timesfm_mse)
                rel_mae_vs_timesfm.setdefault(model_key, []).append(float(metrics["mae"]) / timesfm_mae)

    out: Dict[str, Any] = {}
    for model_key, values in by_model.items():
        item: Dict[str, Any] = {
            "mean_mse": float(np.mean(values["mse"])),
            "mean_mae": float(np.mean(values["mae"])),
            "num_datasets": len(values["mse"]),
        }

        ranks = rank_by_model.get(model_key, [])
        if ranks:
            item["mean_mse_rank"] = float(np.mean(ranks))

        rel_mse_vals = rel_mse_vs_timesfm.get(model_key, [])
        rel_mae_vals = rel_mae_vs_timesfm.get(model_key, [])
        if rel_mse_vals and rel_mae_vals:
            item["mean_rel_mse_vs_timesfm"] = float(np.mean(rel_mse_vals))
            item["mean_rel_mae_vs_timesfm"] = float(np.mean(rel_mae_vals))
            item["num_rel_datasets_vs_timesfm"] = len(rel_mse_vals)

        out[model_key] = item

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TSFM checkpoints and TimesFM baseline on benchmark datasets."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing benchmark csv files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("finetuning_results/benchmark_results.json"),
        help="Path to save evaluation summary json.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_SPECS.keys()),
        default=list(DATASET_SPECS.keys()),
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--strict-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail immediately if any dataset is missing or malformed (default: enabled).",
    )

    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--forecast-horizon", type=int, default=96)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride (default: forecast_horizon).",
    )
    parser.add_argument(
        "--max-test-windows",
        type=int,
        default=256,
        help="Maximum test windows per dataset (None for all).",
    )

    parser.add_argument(
        "--models",
        choices=["both", "tsfm", "timesfm"],
        default="both",
        help="Which models to evaluate.",
    )

    parser.add_argument("--batch-size", type=int, default=32, help="TSFM batch size.")
    parser.add_argument(
        "--timesfm-batch-size",
        type=int,
        default=16,
        help="TimesFM per-call batch size.",
    )

    parser.add_argument("--patch-length", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--pooling",
        choices=["mean", "flatten"],
        default="mean",
    )

    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=[],
        help="TSFM checkpoint files.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default=None,
        help="Glob pattern for TSFM checkpoints (e.g. checkpoints/*.pt).",
    )

    parser.add_argument(
        "--disable-tsfm-input-normalization",
        action="store_true",
        help="Disable TSFM train-stat z-normalization + inverse-scaling path.",
    )
    parser.add_argument(
        "--min-tsfm-load-ratio",
        type=float,
        default=0.9,
        help=(
            "Minimum required fraction of model keys loaded from each checkpoint "
            "(default: 0.9)."
        ),
    )
    parser.add_argument(
        "--allow-missing-forecast-head",
        action="store_true",
        help="Allow evaluation when forecasting_head.* weights are missing (not recommended).",
    )

    parser.add_argument(
        "--timesfm-version",
        choices=["auto", "2p5", "v1"],
        default="auto",
        help="TimesFM API path to use.",
    )
    parser.add_argument(
        "--timesfm-repo-id",
        type=str,
        default=None,
        help="Override HF repo id for TimesFM checkpoint.",
    )
    parser.add_argument(
        "--disable-timesfm-normalize-inputs",
        action="store_true",
        help="Disable TimesFM built-in normalization (not recommended).",
    )
    parser.add_argument(
        "--timesfm-force-cpu",
        action="store_true",
        help="Force CPU backend for legacy TimesFM v1 API.",
    )

    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    if args.max_test_windows is not None and args.max_test_windows <= 0:
        raise ValueError("--max-test-windows must be positive or omitted.")

    if args.context_length <= 0 or args.forecast_horizon <= 0:
        raise ValueError("context_length and forecast_horizon must be positive.")
    if not (0.0 <= args.min_tsfm_load_ratio <= 1.0):
        raise ValueError("--min-tsfm-load-ratio must be between 0 and 1.")

    use_tsfm = args.models in ("both", "tsfm")
    use_timesfm = args.models in ("both", "timesfm")

    tsfm_ckpts: List[Path] = []
    if use_tsfm:
        tsfm_ckpts = _resolve_checkpoints(args.checkpoints, args.checkpoint_glob)
        if not tsfm_ckpts and Path("tsfm_pretrain.pt").exists():
            tsfm_ckpts = [Path("tsfm_pretrain.pt")]

        if not tsfm_ckpts:
            raise ValueError(
                "TSFM evaluation selected but no checkpoints were found. "
                "Use --checkpoints and/or --checkpoint-glob."
            )

    payloads = _prepare_payloads(args)
    if not payloads:
        raise RuntimeError("No datasets available for evaluation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Prepared {len(payloads)} dataset(s) for evaluation")

    dataset_results: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        dataset_results[payload.name] = {
            "csv_path": str(payload.csv_path),
            "target_column": payload.target_column,
            "freq_bucket": payload.freq_bucket,
            "split": payload.split_info,
            "num_windows": int(payload.contexts.shape[0]),
            "models": {},
        }

    # TimesFM baseline first.
    if use_timesfm:
        print("\n[INFO] Initializing TimesFM baseline...")
        timesfm_runner = TimesFMBaseline(
            version=args.timesfm_version,
            repo_id=args.timesfm_repo_id,
            horizon=args.forecast_horizon,
            max_context=args.context_length,
            batch_size=args.timesfm_batch_size,
            normalize_inputs=(not args.disable_timesfm_normalize_inputs),
            force_cpu=args.timesfm_force_cpu,
        )
        print(f"[INFO] TimesFM implementation: {timesfm_runner.impl}")

        for payload in payloads:
            print(f"[INFO] TimesFM -> {payload.name} ({payload.contexts.shape[0]} windows)")
            pred = timesfm_runner.predict(payload.contexts, freq_bucket=payload.freq_bucket)
            metrics = _compute_metrics(pred, payload.targets)
            dataset_results[payload.name]["models"]["timesfm"] = {
                **metrics,
                "impl": timesfm_runner.impl,
                "normalize_inputs": (not args.disable_timesfm_normalize_inputs),
            }

    # TSFM checkpoints.
    if use_tsfm:
        for ckpt in tsfm_ckpts:
            print(f"\n[INFO] Loading TSFM checkpoint: {ckpt}")
            model, load_meta = _load_tsfm_model(
                checkpoint_path=ckpt,
                device=device,
                context_length=args.context_length,
                patch_length=args.patch_length,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                forecast_horizon=args.forecast_horizon,
                pooling=args.pooling,
            )

            label = f"tsfm::{ckpt.name}"

            if load_meta["missing_keys"]:
                print(
                    f"[WARN] Missing keys for {ckpt.name}: "
                    f"{len(load_meta['missing_keys'])}"
                )
            if load_meta["unexpected_keys"]:
                print(
                    f"[WARN] Unexpected keys for {ckpt.name}: "
                    f"{len(load_meta['unexpected_keys'])}"
                )

            missing_forecast_head = [
                k
                for k in load_meta.get("missing_keys", [])
                if k.startswith("forecasting_head.")
            ]
            if missing_forecast_head and (not args.allow_missing_forecast_head):
                raise RuntimeError(
                    f"Checkpoint {ckpt.name} is missing forecasting head weights: "
                    f"{missing_forecast_head}. This is usually a pretraining checkpoint and "
                    "not suitable for forecasting evaluation."
                )

            quality = load_meta.get("quality", {})
            load_ratio = float(quality.get("load_ratio", 0.0))
            print(
                f"[INFO] Load quality for {ckpt.name}: "
                f"loaded={quality.get('loaded_keys', 0)}/{quality.get('total_model_keys', 0)} "
                f"ratio={load_ratio:.3f}"
            )
            if load_ratio < args.min_tsfm_load_ratio:
                raise RuntimeError(
                    f"Checkpoint load quality too low for {ckpt.name}: "
                    f"ratio={load_ratio:.3f} < threshold={args.min_tsfm_load_ratio:.3f}. "
                    "Likely state_dict/model mismatch."
                )

            inferred_arch = load_meta.get("inferred_architecture", {})
            model_horizon = int(inferred_arch.get("model_forecast_horizon", args.forecast_horizon))
            if model_horizon < args.forecast_horizon:
                print(
                    f"[INFO] Using autoregressive rollout for {ckpt.name}: "
                    f"model_horizon={model_horizon} -> eval_horizon={args.forecast_horizon}"
                )

            for payload in payloads:
                print(f"[INFO] TSFM {ckpt.name} -> {payload.name} ({payload.contexts.shape[0]} windows)")

                if args.disable_tsfm_input_normalization:
                    tsfm_contexts = payload.contexts
                else:
                    tsfm_contexts = (payload.contexts - payload.train_mean) / payload.train_std

                pred = _predict_tsfm_with_rollout(
                    model=model,
                    contexts=tsfm_contexts,
                    target_horizon=args.forecast_horizon,
                    model_horizon=model_horizon,
                    batch_size=args.batch_size,
                    device=device,
                )

                # Reconcile TSFM output scale back to raw dataset units.
                if args.disable_tsfm_input_normalization:
                    pred_raw = pred
                else:
                    pred_raw = pred * payload.train_std + payload.train_mean

                metrics = _compute_metrics(pred_raw, payload.targets)
                dataset_results[payload.name]["models"][label] = {
                    **metrics,
                    "checkpoint": str(ckpt),
                    "tsfm_input_normalized": (not args.disable_tsfm_input_normalization),
                    "train_mean": payload.train_mean,
                    "train_std": payload.train_std,
                    "load_meta": load_meta,
                }

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate = _aggregate_metrics(dataset_results)

    output = {
        "config": {
            "datasets": args.datasets,
            "context_length": args.context_length,
            "forecast_horizon": args.forecast_horizon,
            "stride": args.stride if args.stride is not None else args.forecast_horizon,
            "max_test_windows": args.max_test_windows,
            "models": args.models,
        },
        "scaling_reconciliation": {
            "timesfm_internal_normalization": (not args.disable_timesfm_normalize_inputs),
            "tsfm_train_stat_normalization": (not args.disable_tsfm_input_normalization),
            "metric_domain": "raw_units",
        },
        "datasets": dataset_results,
        "aggregate": aggregate,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for model_key, stats in aggregate.items():
        line = (
            f"{model_key:45s} "
            f"mean MSE={stats['mean_mse']:.6f} "
            f"mean MAE={stats['mean_mae']:.6f} "
            f"datasets={stats['num_datasets']}"
        )
        if "mean_mse_rank" in stats:
            line += f" mean-rank={stats['mean_mse_rank']:.3f}"
        if "mean_rel_mse_vs_timesfm" in stats:
            line += (
                f" relMSE-vs-timesfm={stats['mean_rel_mse_vs_timesfm']:.3f}"
                f" relMAE-vs-timesfm={stats['mean_rel_mae_vs_timesfm']:.3f}"
            )
        print(line)
    print(f"\n[INFO] Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
