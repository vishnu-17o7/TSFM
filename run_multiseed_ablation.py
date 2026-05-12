"""
Multi-seed TSFM ablation runner with statistical rigour.

Key improvements over the original:
- Sufficient training budget (10 epochs by default, full dataset passes)
- Validation-based early stopping with best-checkpoint saving
- Per-epoch train + val loss tracking for convergence analysis
- Increased seed count (7 seeds default, up from 3)
- Bootstrap confidence intervals (alongside Student-t)
- Cohen's d effect sizes vs default config
- Seed-count planning mode for estimating required n
- Real-world data support via --real-data-dir
"""

import argparse
import copy
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from train_tsfm import (
    EMA,
    TSFMPretrain,
    WindowDataset,
    _as_jsonable,
    _write_metrics,
    get_cosine_schedule_with_warmup,
    load_all_series,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic multi-seed TSFM ablations with validation and convergence tracking."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--real-data-dir", type=Path, default=None,
                        help="Directory with real .tsf/.ts pretraining corpora (loaded alongside synthetic data).")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/multiseed"))
    parser.add_argument("--summary-csv", type=Path, default=Path("experiments/multiseed_summary.csv"))
    parser.add_argument("--seeds", type=str, default="11,42,123,256,512,789,1024")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0,
                        help="Maximum steps per epoch (0 = full dataset pass).")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Stop training if val loss does not improve for N epochs.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5,
                        help="Minimum improvement in val loss to count as progress.")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of the dataset to use for validation.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic-length", type=int, default=1024)
    parser.add_argument("--progress-every-rows", type=int, default=0)
    parser.add_argument("--feature-workers", type=int, default=1)
    parser.add_argument("--max-rows-per-feature-file", type=int, default=50)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ci-method", choices=["t", "bootstrap", "both"], default="both",
                        help="Confidence interval method for summary.")
    parser.add_argument("--bootstrap-samples", type=int, default=10000,
                        help="Number of bootstrap resamples for CI estimation.")
    parser.add_argument("--estimate-seeds", action="store_true",
                        help="Run a pilot (first 3 seeds only) and estimate required seed count for target precision.")
    parser.add_argument("--target-ci-half-width-ratio", type=float, default=0.10,
                        help="Target CI half-width as a fraction of the mean (for --estimate-seeds).")
    parser.add_argument("--real-data-only", action="store_true",
                        help="Skip synthetic feature fallback; use only real corpora from --real-data-dir.")
    parser.add_argument("--loss-fn", type=str, choices=["mse", "huber"], default="huber",
                        help="Loss function (huber is more robust to outliers).")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate for weight averaging (0 = disabled).")
    parser.add_argument("--no-compile", action="store_true", default=False,
                        help="Disable torch.compile (PyTorch 2.0+).")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False,
                        help="Use gradient checkpointing to save memory.")
    return parser.parse_args()


def _parse_seed_list(raw: str) -> List[int]:
    out: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("At least one seed is required")
    return out


def _build_ablation_grid(include_data_ablation: bool = False) -> List[Dict]:
    base = [
        {"label": "mask_0p2", "mask_ratio": 0.2, "patch_length": 16},
        {"label": "mask_0p4", "mask_ratio": 0.4, "patch_length": 16},
        {"label": "mask_0p6", "mask_ratio": 0.6, "patch_length": 16},
        {"label": "patch_8", "mask_ratio": 0.4, "patch_length": 8},
        {"label": "patch_32", "mask_ratio": 0.4, "patch_length": 32},
    ]
    if include_data_ablation:
        base.append({"label": "real_data", "mask_ratio": 0.4, "patch_length": 16, "real_data_only": True})
    return base


def _validate_model(model: TSFMPretrain, val_loader: DataLoader, device: torch.device,
                    mask_ratio: float, use_amp: bool, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp):
                reconstructed, original, mask = model(batch, mask_ratio=mask_ratio)
                if not mask.any():
                    continue
                preds = reconstructed[mask]
                targets = original[mask]
                loss = criterion(preds, targets)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def _run_single(
    *,
    cfg: Dict,
    seed: int,
    data_seed: int,
    args: argparse.Namespace,
    train_dataset,
    val_dataset,
    data_meta: Dict[str, object],
    cpu_count: int,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, object]:
    set_seed(seed)
    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed)

    model = TSFMPretrain(
        context_length=args.context_length,
        patch_length=int(cfg["patch_length"]),
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
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=use_amp)

    # --- Loss function selection ---
    if args.loss_fn == "huber":
        criterion = nn.SmoothL1Loss(beta=1.0)
    else:
        criterion = nn.MSELoss()

    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        generator=torch_gen,
    )
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_dataset, **dl_kwargs)

    val_dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_dataset, **val_dl_kwargs) if val_dataset is not None else None

    # --- Learning rate scheduler ---
    total_batches = len(train_loader)
    steps_per_epoch = total_batches
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(args.max_steps_per_epoch, total_batches)
    total_training_steps = args.epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)

    # --- EMA initialization ---
    ema = None
    if args.ema_decay > 0:
        ema = EMA(model, decay=args.ema_decay)

    run_name = f"ms_{cfg['label']}_s{seed}"
    started_at = datetime.now().isoformat(timespec="seconds")
    run_start = time.perf_counter()

    metrics_payload: Dict[str, object] = {
        "run_name": run_name,
        "started_at": started_at,
        "finished_at": None,
        "config": _as_jsonable(
            {
                "seed": seed,
                "data_seed": data_seed,
                "epochs": args.epochs,
                "max_steps_per_epoch": args.max_steps_per_epoch,
                "early_stopping_patience": args.early_stopping_patience,
                "batch_size": args.batch_size,
                "context_length": args.context_length,
                "stride": args.stride,
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "num_workers": args.num_workers,
                "feature_fallback": args.feature_fallback,
                "real_data_only": cfg.get("real_data_only", False),
                "synthetic_length": args.synthetic_length,
                "max_rows_per_feature_file": args.max_rows_per_feature_file,
                "mask_ratio": cfg["mask_ratio"],
                "patch_length": int(cfg["patch_length"]),
                "loss_fn": args.loss_fn,
                "ema_decay": args.ema_decay,
            }
        ),
        "system": {
            "device": str(device),
            "cpu_count": cpu_count,
            "use_amp": bool(use_amp),
        },
        "data": {
            "source_mode": data_meta["source_mode"],
            "loaded_files": data_meta["loaded_files"],
            "loaded_series": data_meta["loaded_series"],
            "total_windows": len(train_dataset),
            "val_windows": len(val_dataset) if val_dataset is not None else 0,
            "real_ratio": data_meta.get("real_ratio", 0.0),
        },
        "model": {
            "total_params": int(sum(p.numel() for p in model.parameters())),
        },
        "epochs": [],
        "summary": {},
    }

    accum_steps = max(1, int(args.gradient_accumulation_steps))
    mask_ratio = float(cfg["mask_ratio"])

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        step_count = 0
        micro_step = 0

        max_steps = total_batches
        if args.max_steps_per_epoch > 0:
            max_steps = min(args.max_steps_per_epoch, total_batches)

        for batch in train_loader:
            if args.max_steps_per_epoch > 0 and step_count >= args.max_steps_per_epoch:
                break

            batch = batch.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                reconstructed, original, mask = model(batch, mask_ratio=mask_ratio)
                if not mask.any():
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

        if micro_step % accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_train_loss = running_loss / max(step_count, 1)
        epoch_time_sec = time.perf_counter() - epoch_start

        # Track GPU peak memory
        gpu_peak_mb = None
        if device.type == "cuda":
            gpu_peak_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
            torch.cuda.reset_peak_memory_stats()

        epoch_metrics = {
            "epoch": epoch,
            "steps": int(step_count),
            "train_masked_mse": float(avg_train_loss),
            "epoch_time_sec": float(epoch_time_sec),
        }
        if gpu_peak_mb is not None:
            epoch_metrics["gpu_peak_mb"] = gpu_peak_mb

        if val_loader is not None:
            val_loss = _validate_model(model, val_loader, device, mask_ratio, use_amp, criterion)
            epoch_metrics["val_masked_mse"] = float(val_loss)

            # Also evaluate EMA if enabled
            ema_val_loss = None
            if ema is not None:
                original_state = copy.deepcopy(model.state_dict())
                ema.apply_to(model)
                ema_val_loss = _validate_model(model, val_loader, device, mask_ratio, use_amp, criterion)
                model.load_state_dict(original_state)
                epoch_metrics["ema_val_masked_mse"] = float(ema_val_loss)

            effective_val = val_loss
            if ema_val_loss is not None and ema_val_loss < val_loss:
                effective_val = ema_val_loss

            if effective_val < best_val_loss - args.early_stopping_min_delta:
                best_val_loss = float(effective_val)
                best_epoch = epoch
                epochs_without_improvement = 0
                if ema is not None and ema_val_loss is not None and ema_val_loss < val_loss:
                    best_state_dict = ema.state_dict()
                else:
                    best_state_dict = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

        metrics_payload["epochs"].append(epoch_metrics)

        if val_loader is not None and epochs_without_improvement >= args.early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} (best val={best_val_loss:.6f} at epoch {best_epoch})")
            break

    elapsed_sec = time.perf_counter() - run_start

    num_epochs = len(metrics_payload["epochs"])
    train_losses = [ep["train_masked_mse"] for ep in metrics_payload["epochs"]]
    val_losses = [ep.get("val_masked_mse") for ep in metrics_payload["epochs"] if ep.get("val_masked_mse") is not None]

    final_train_loss = float(train_losses[-1]) if train_losses else None
    best_train_loss = float(min(train_losses)) if train_losses else None
    final_val_loss = float(val_losses[-1]) if val_losses else None
    best_val_loss_final = float(min(val_losses)) if val_losses else None

    delta_last_n = None
    if num_epochs >= 4:
        recent = train_losses[-3:]
        delta_last_n = float(recent[0] - recent[-1])

    checkpoint_path = None
    if args.save_checkpoints and best_state_dict is not None:
        ckpt_dir = args.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / f"{run_name}_best.pt"
        torch.save({
            "model_state_dict": best_state_dict,
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "config": metrics_payload["config"],
        }, checkpoint_path)
        checkpoint_path = str(checkpoint_path)

    metrics_payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
    metrics_payload["summary"] = {
        "total_train_time_sec": float(elapsed_sec),
        "best_train_mse": best_train_loss,
        "final_train_mse": final_train_loss,
        "best_val_mse": best_val_loss_final,
        "final_val_mse": final_val_loss,
        "best_epoch": best_epoch,
        "num_epochs_completed": num_epochs,
        "delta_last_n": delta_last_n,
        "converged": bool(delta_last_n is not None and abs(delta_last_n) < args.early_stopping_min_delta),
        "checkpoint_path": checkpoint_path,
    }

    return metrics_payload


def _bootstrap_ci(values: np.ndarray, n_samples: int, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_samples):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    means = np.array(means)
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    mean_a, mean_b = float(np.mean(group_a)), float(np.mean(group_b))
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (mean_a - mean_b) / pooled_std


def _summary_rows(
    run_payloads: List[Dict[str, object]],
    batch_size: int,
    context_length: int,
    ci_method: str = "both",
    bootstrap_samples: int = 10000,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    try:
        from scipy import stats as scipy_stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    flat: List[Dict[str, object]] = []
    for payload in run_payloads:
        cfg = payload["config"]
        last_epoch = payload["epochs"][-1]
        patch = int(cfg["patch_length"])
        steps = int(last_epoch["steps"])
        epoch_time = float(last_epoch["epoch_time_sec"])
        token_count = context_length // patch
        token_throughput = (steps * batch_size * token_count) / max(epoch_time, 1e-8)
        summary = payload.get("summary", {})
        flat.append({
            "label": payload["run_name"].split("_s")[0].replace("ms_", ""),
            "seed": int(cfg["seed"]),
            "mask_ratio": float(cfg["mask_ratio"]),
            "patch_length": patch,
            "real_data_only": bool(cfg.get("real_data_only", False)),
            "steps": steps,
            "train_mse": float(last_epoch.get("train_masked_mse", last_epoch.get("masked_mse", 0))),
            "val_mse": float(last_epoch.get("val_masked_mse", None)) if last_epoch.get("val_masked_mse") is not None else None,
            "best_train_mse": float(summary.get("best_train_mse", 0)) if summary.get("best_train_mse") is not None else None,
            "best_val_mse": float(summary.get("best_val_mse", 0)) if summary.get("best_val_mse") is not None else None,
            "epochs_completed": int(summary.get("num_epochs_completed", 0)),
            "converged": bool(summary.get("converged", False)),
            "delta_last_n": float(summary.get("delta_last_n", 0)) if summary.get("delta_last_n") is not None else None,
            "epoch_time_sec": epoch_time,
            "token_throughput": token_throughput,
            "total_train_time_sec": float(summary.get("total_train_time_sec", 0)),
        })

    df = pd.DataFrame(flat)
    metric_col = "best_train_mse"
    val_col = "best_val_mse"

    group_cols = ["label", "mask_ratio", "patch_length"]
    grouped = df.groupby(group_cols, as_index=False)

    default_values = {}
    for _, g in grouped:
        label = str(g["label"].iloc[0])
        if label == "mask_0p4":
            default_values["mask_0p4"] = g[metric_col].values

    for _, g in grouped:
        n = int(len(g))
        label = str(g["label"].iloc[0])
        mse_values = g[metric_col].values.astype(float)
        mse_mean = float(np.mean(mse_values))
        mse_std = float(np.std(mse_values, ddof=1)) if n > 1 else 0.0
        sem = mse_std / math.sqrt(n) if n > 1 else 0.0

        ci95_lower = None
        ci95_upper = None
        ci_method_used = "none"

        if n >= 2:
            if ci_method in ("t", "both") and HAS_SCIPY:
                try:
                    t_val = scipy_stats.t.ppf(0.975, df=n - 1)
                    ci95_lower = mse_mean - t_val * sem
                    ci95_upper = mse_mean + t_val * sem
                    ci_method_used = "t"
                except Exception:
                    pass
            elif ci_method in ("t", "both"):
                t_critical_lut = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
                                  8: 2.365, 9: 2.306, 10: 2.262}
                t_val = t_critical_lut.get(n, 1.96)
                ci95_lower = mse_mean - t_val * sem
                ci95_upper = mse_mean + t_val * sem
                ci_method_used = "t (lut)"

            if ci_method in ("bootstrap", "both") and n >= 3:
                try:
                    b_lower, b_upper = _bootstrap_ci(mse_values, bootstrap_samples)
                    if ci_method == "bootstrap":
                        ci95_lower = b_lower
                        ci95_upper = b_upper
                        ci_method_used = "bootstrap"
                    else:
                        ci_method_used = "both"
                except Exception:
                    pass

        significant = None
        if ci95_lower is not None and ci95_upper is not None:
            significant = bool(ci95_lower > 0 or ci95_upper < 0)

        cohens_d_vs_default = None
        welch_pvalue_vs_default = None
        if "mask_0p4" in default_values and label != "mask_0p4":
            cohens_d_vs_default = _cohens_d(mse_values, default_values["mask_0p4"])
            # Welch's t-test (unequal variance) for pairwise comparison
            if HAS_SCIPY and n >= 2 and len(default_values["mask_0p4"]) >= 2:
                try:
                    _, p_val = scipy_stats.ttest_ind(mse_values, default_values["mask_0p4"], equal_var=False)
                    welch_pvalue_vs_default = float(p_val)
                except Exception:
                    pass

        val_mean = float(np.mean(g[val_col].dropna())) if not g[val_col].isna().all() else None
        time_mean = float(g["epoch_time_sec"].mean())
        thr_mean = float(g["token_throughput"].mean())
        converged_frac = float(g["converged"].mean())
        epochs_mean = float(g["epochs_completed"].mean())

        row = {
            "label": label,
            "mask_ratio": float(g["mask_ratio"].iloc[0]),
            "patch_length": int(g["patch_length"].iloc[0]),
            "n_seeds": n,
            "steps_mean": float(g["steps"].mean()),
            "train_mse_mean": mse_mean,
            "train_mse_std": mse_std,
            "train_mse_sem": sem,
            "train_mse_ci95_lower": ci95_lower,
            "train_mse_ci95_upper": ci95_upper,
            "ci_method": ci_method_used,
            "significant_at_p05": significant,
            "cohens_d_vs_default": cohens_d_vs_default,
            "welch_pvalue_vs_default": welch_pvalue_vs_default,
            "val_mse_mean": val_mean,
            "epochs_mean": epochs_mean,
            "converged_fraction": converged_frac,
            "epoch_time_mean_sec": time_mean,
            "token_throughput_mean": thr_mean,
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(by=["patch_length", "mask_ratio"]).reset_index(drop=True)

    # --- Holm-Bonferroni multiple comparison correction ---
    if HAS_SCIPY and "welch_pvalue_vs_default" in out.columns:
        raw_pvals = out["welch_pvalue_vs_default"].dropna().values
        if len(raw_pvals) > 1:
            # Holm-Bonferroni step-down correction
            n_tests = len(raw_pvals)
            sorted_indices = np.argsort(raw_pvals)
            corrected = np.zeros_like(raw_pvals)
            for rank, idx in enumerate(sorted_indices):
                corrected[idx] = min(1.0, raw_pvals[idx] * (n_tests - rank))
            # Enforce monotonicity
            running_max = 0.0
            for idx in sorted_indices:
                corrected[idx] = max(corrected[idx], running_max)
                running_max = corrected[idx]
            # Map back to dataframe
            pval_idx = out["welch_pvalue_vs_default"].dropna().index
            out.loc[pval_idx, "holm_corrected_pvalue"] = corrected
            out["significant_holm_p05"] = out["holm_corrected_pvalue"].apply(
                lambda x: bool(x < 0.05) if pd.notna(x) else None
            )

    return out


def _estimate_seeds_needed(
    pilot_payloads: List[Dict[str, object]],
    target_half_width_ratio: float,
) -> pd.DataFrame:
    try:
        from scipy import stats as scipy_stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    flat: List[Dict] = []
    for payload in pilot_payloads:
        cfg = payload["config"]
        last_epoch = payload["epochs"][-1]
        flat.append({
            "label": payload["run_name"].split("_s")[0].replace("ms_", ""),
            "seed": int(cfg["seed"]),
            "mask_ratio": float(cfg["mask_ratio"]),
            "patch_length": int(cfg["patch_length"]),
            "train_mse": float(last_epoch.get("train_masked_mse", last_epoch.get("masked_mse", 0))),
        })

    df = pd.DataFrame(flat)
    group_cols = ["label", "mask_ratio", "patch_length"]

    rows = []
    for _, g in df.groupby(group_cols, as_index=False):
        n_pilot = int(len(g))
        mse_values = g["train_mse"].values.astype(float)
        mean_val = float(np.mean(mse_values))
        std_val = float(np.std(mse_values, ddof=1)) if n_pilot > 1 else 0.0

        target_half_width = target_half_width_ratio * abs(mean_val) if abs(mean_val) > 1e-12 else target_half_width_ratio
        if std_val < 1e-12:
            n_required = n_pilot
        else:
            if HAS_SCIPY:
                n_required = int(math.ceil((scipy_stats.t.ppf(0.975, df=n_pilot - 1) * std_val / target_half_width) ** 2))
            else:
                n_required = int(math.ceil((1.96 * std_val / target_half_width) ** 2))
        n_required = max(n_pilot, n_required)

        rows.append({
            "label": str(g["label"].iloc[0]),
            "mask_ratio": float(g["mask_ratio"].iloc[0]),
            "patch_length": int(g["patch_length"].iloc[0]),
            "n_pilot": n_pilot,
            "mse_mean": mean_val,
            "mse_std": std_val,
            "target_half_width": target_half_width,
            "n_required_estimated": n_required,
        })

    return pd.DataFrame(rows).sort_values(by=["patch_length", "mask_ratio"]).reset_index(drop=True)


def _load_data_for_seed(
    args: argparse.Namespace,
    seed: int,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    series_list: List[np.ndarray] = []
    source_modes: List[str] = []
    total_loaded_files = 0
    synthetic_series_count = 0
    real_series_count = 0

    if args.real_data_only:
        args.feature_fallback = False

    if not args.real_data_only:
        syn_series, syn_files, syn_mode = load_all_series(
            args.data_dir,
            feature_fallback=args.feature_fallback,
            synthetic_length=args.synthetic_length,
            seed=seed,
            progress_every_rows=args.progress_every_rows,
            feature_workers=args.feature_workers,
            max_rows_per_feature_file=args.max_rows_per_feature_file,
            exclude_dir=args.real_data_dir,
        )
        series_list.extend(syn_series)
        source_modes.append(syn_mode)
        total_loaded_files += len(syn_files)
        synthetic_series_count += len(syn_series)

    if args.real_data_dir is not None and args.real_data_dir.exists():
        real_series, real_files, real_mode = load_all_series(
            args.real_data_dir,
            feature_fallback=False,
            synthetic_length=args.synthetic_length,
            seed=seed,
            progress_every_rows=args.progress_every_rows,
            feature_workers=args.feature_workers,
            max_rows_per_feature_file=0,
            exclude_dir=None,
        )
        series_list.extend(real_series)
        source_modes.append(real_mode)
        total_loaded_files += len(real_files)
        real_series_count += len(real_series)

    if not series_list:
        raise RuntimeError("No series loaded. Check --data-dir and --real-data-dir paths.")

    if "raw" in source_modes and "feature-fallback" in source_modes:
        source_mode = "mixed"
    elif "raw" in source_modes:
        source_mode = "raw"
    elif "feature-fallback" in source_modes:
        source_mode = "feature-fallback"
    else:
        source_mode = source_modes[0]

    total_series = synthetic_series_count + real_series_count
    real_ratio = real_series_count / max(total_series, 1) if total_series > 0 else 0.0

    data_meta = {
        "source_mode": source_mode,
        "loaded_files": total_loaded_files,
        "loaded_series": len(series_list),
        "synthetic_series": synthetic_series_count,
        "real_series": real_series_count,
        "real_ratio": real_ratio,
    }

    return series_list, data_meta


def main() -> None:
    args = parse_args()
    seeds = _parse_seed_list(args.seeds)
    ablations = _build_ablation_grid()

    if args.estimate_seeds and len(seeds) < 3:
        print("[INFO] --estimate-seeds requires at least 3 seeds for a pilot. Resetting to first 3.")
        seeds = seeds[:3]

    cpu_count = os.cpu_count() or 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    print(f"Device: {device}")
    print(f"CPU cores: {cpu_count}")
    print(f"Seeds: {seeds} ({len(seeds)} seeds)")
    print(f"Epochs: {args.epochs} | Max steps/epoch: {args.max_steps_per_epoch or 'full pass'}")
    print(f"Early stopping: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")
    print(f"CI method: {args.ci_method}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_payloads: List[Dict[str, object]] = []

    for seed in seeds:
        print(f"\n[DATA] Loading corpus with seed={seed}")
        series_list, data_meta = _load_data_for_seed(args, seed)

        full_dataset = WindowDataset(series_list, context_length=args.context_length, stride=args.stride)

        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        if train_size < 1:
            raise RuntimeError(f"Not enough windows for training (train_size={train_size}). Reduce val_split or increase data.")

        if val_size > 0:
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(seed),
            )
        else:
            train_dataset = full_dataset
            val_dataset = None

        print(f"  Train windows: {train_size}, Val windows: {val_size}")

        for cfg in ablations:
            print(f"[RUN] {cfg['label']} seed={seed}")
            payload = _run_single(
                cfg=cfg,
                seed=seed,
                data_seed=seed,
                args=args,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                data_meta=data_meta,
                cpu_count=cpu_count,
                device=device,
                use_amp=use_amp,
            )
            run_payloads.append(payload)

            if args.save_json:
                out_path = args.output_dir / f"{cfg['label']}_seed{seed}.json"
                _write_metrics(out_path, payload)

            summary = payload["summary"]
            best_train = summary.get("best_train_mse")
            best_val = summary.get("best_val_mse")
            elapsed = summary["total_train_time_sec"]
            n_epochs = summary["num_epochs_completed"]
            msg = f"[DONE] {cfg['label']} seed={seed} | train_mse={best_train:.6f}"
            if best_val is not None:
                msg += f" | val_mse={best_val:.6f}"
            msg += f" | epochs={n_epochs} | time={elapsed:.2f}s"
            print(msg)

    if args.estimate_seeds:
        plan_df = _estimate_seeds_needed(run_payloads, args.target_ci_half_width_ratio)
        plan_path = args.output_dir / "seed_count_plan.csv"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_df.to_csv(plan_path, index=False)
        print(f"\n[INFO] Seed-count plan saved to {plan_path}")
        print(plan_df.to_string(index=False))

    summary_df = _summary_rows(
        run_payloads,
        batch_size=args.batch_size,
        context_length=args.context_length,
        ci_method=args.ci_method,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_csv, index=False)
    print(f"\nSaved summary CSV: {args.summary_csv}")

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    cols = ["label", "n_seeds", "train_mse_mean", "train_mse_std", "val_mse_mean",
            "cohens_d_vs_default", "welch_pvalue_vs_default", "significant_holm_p05",
            "significant_at_p05", "converged_fraction"]
    available = [c for c in cols if c in summary_df.columns]
    print(summary_df[available].to_string(index=False))


if __name__ == "__main__":
    main()
