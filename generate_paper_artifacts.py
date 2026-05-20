"""
Generate all LaTeX table snippets and publication figures for paper_tsfm_ieee.tex.

Reads from:
  - experiments/ablation_summary_det.csv
  - experiments/multiseed_summary.csv
  - experiments/multiseed_runs.csv
  - experiments/multiseed_effects.csv
  - experiments/checkpoint_eval/final_checkpoint_eval.csv
  - experiments/checkpoint_eval/final_checkpoint_eval_summary.json
  - finetuning_results/results.json
  - finetuning_results/benchmark_results.json

Outputs:
  - paper/tables/*.tex          (LaTeX table bodies, ready for \\input{})
  - paper/fig_*.png             (Publication-quality figures)
  - paper/paper_values.json     (All scalar values for manual verification)

Usage:
  python generate_paper_artifacts.py [--paper-dir paper] [--skip-figures]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Utility helpers ─────────────────────────────────────────────────────────

def _safe_read_csv(path: Path) -> Optional["pd.DataFrame"]:
    if not HAS_PANDAS:
        print(f"[ERROR] pandas is required. pip install pandas")
        return None
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return None
    return pd.read_csv(path)


def _safe_read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_tex(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [OK] {path}")


def _fmt(val, decimals=4):
    """Format a float for LaTeX tables."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "---"
    return f"{val:.{decimals}f}"


def _pct(val, decimals=2):
    if val is None:
        return "---"
    return f"{val:+.{decimals}f}\\%"


# ── Table generators ────────────────────────────────────────────────────────

def gen_tab_mask_ablation(det_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 4: Single-Seed Mask-Ratio Ablation (Patch Length = 16)"""
    rows = det_df[det_df["patch_length"] == 16].sort_values("mask_ratio")
    lines = []
    for _, r in rows.iterrows():
        mse_val = float(r["masked_mse"])
        # Handle catastrophic values
        if mse_val > 1e6:
            mse_str = f"${mse_val:.2e}$".replace("e+", r"\\times10^{") + "}"
        else:
            mse_str = _fmt(mse_val)
        lines.append(
            f"{r['mask_ratio']:.1f} & {int(r['steps'])} & {mse_str} & {_fmt(r['epoch_time_sec'])}"
        )
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "Mask Ratio $\\rho$ & Steps & Masked MSE & Epoch Time (s) \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_mask_ablation.tex", tex)
    values["tab_mask_ablation"] = rows[["mask_ratio", "steps", "masked_mse", "epoch_time_sec"]].to_dict("records")


def gen_tab_patch_ablation(det_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 5: Single-Seed Patch-Length Ablation (Mask Ratio = 0.4)"""
    rows = det_df[det_df["mask_ratio"] == 0.4].sort_values("patch_length")
    lines = []
    for _, r in rows.iterrows():
        pl = int(r["patch_length"])
        N = 512 // pl
        mse_val = float(r["masked_mse"])
        if mse_val > 1e6:
            mse_str = f"${mse_val:.2e}$".replace("e+", r"\\times10^{") + "}"
        else:
            mse_str = _fmt(mse_val)
        # Compute token throughput
        steps = int(r["steps"])
        batch = 32
        epoch_time = float(r["epoch_time_sec"])
        tput = (steps * batch * N) / epoch_time if epoch_time > 0 else 0
        lines.append(
            f"{pl} & {N} & {steps} & {mse_str} & {_fmt(epoch_time)} & {_fmt(tput, 2)}"
        )
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "Patch $P$ & Tokens $N$ & Steps & Masked MSE & Epoch Time (s) & Token/s \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_patch_ablation.tex", tex)


def gen_tab_mask_multiseed(ms_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 6: Multi-Seed Mask-Ratio Results"""
    rows = ms_df[ms_df["patch_length"] == 16].sort_values("mask_ratio")
    lines = []
    for _, r in rows.iterrows():
        lines.append(
            f"{r['mask_ratio']:.1f} & {_fmt(r['masked_mse_mean'])} & "
            f"{_fmt(r['masked_mse_std'])} & {_fmt(r['masked_mse_ci95'])} & "
            f"{_fmt(r['epoch_time_mean_sec'])}"
        )
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "$\\rho$ & Mean MSE & Std & CI$_{95}$ HW & Time (s) \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_mask_multiseed.tex", tex)
    values["tab_mask_multiseed"] = rows[
        ["mask_ratio", "masked_mse_mean", "masked_mse_std", "masked_mse_ci95"]
    ].to_dict("records")


def gen_tab_patch_multiseed(ms_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 7: Multi-Seed Patch-Length Results"""
    rows = ms_df[ms_df["mask_ratio"] == 0.4].sort_values("patch_length")
    lines = []
    for _, r in rows.iterrows():
        pl = int(r["patch_length"])
        N = 512 // pl
        lines.append(
            f"{pl} & {N} & {_fmt(r['masked_mse_mean'])} & "
            f"{_fmt(r['masked_mse_ci95'])} & {_fmt(r['epoch_time_mean_sec'])} & "
            f"{_fmt(r['token_throughput_mean'], 2)}"
        )
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "$P$ & $N$ & Mean MSE & CI$_{95}$ HW & Time (s) & Token/s \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_patch_multiseed.tex", tex)
    values["tab_patch_multiseed"] = rows[
        ["patch_length", "masked_mse_mean", "masked_mse_ci95", "epoch_time_mean_sec", "token_throughput_mean"]
    ].to_dict("records")


def gen_tab_effects(eff_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 8: Relative Effects vs Baseline"""
    baseline_label = "mask_0p4"
    lines = []
    for _, r in eff_df.iterrows():
        if r["label"] == baseline_label:
            continue
        label_parts = []
        mr = float(r["mask_ratio"])
        pl = int(r["patch_length"])
        setting = f"$\\rho={mr}$, $P={pl}$"
        lines.append(
            f"{setting} & ${_pct(r['mse_pct_vs_baseline'])}$ & {_fmt(r['time_ratio_vs_baseline'], 2)}"
        )
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "Setting & $\\Delta$MSE (\\%) & Time Ratio \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_effects.tex", tex)


def gen_tab_frontier_metrics(ms_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 9: Derived Stability and Efficiency Metrics"""
    lines = []
    records = []
    for _, r in ms_df.sort_values(["patch_length", "mask_ratio"]).iterrows():
        mean_mse = float(r["masked_mse_mean"])
        std_mse = float(r["masked_mse_std"])
        mean_time = float(r["epoch_time_mean_sec"])
        cv = std_mse / mean_mse if mean_mse > 0 else 0
        sei = 1.0 / (mean_mse * mean_time) if (mean_mse * mean_time) > 0 else 0
        pl = int(r["patch_length"])
        mr = float(r["mask_ratio"])
        setting = f"$P={pl},\\rho={mr}$"
        lines.append(
            f"{setting} & {_fmt(mean_mse)} & {_fmt(cv)} & {_fmt(sei)}"
        )
        records.append({"patch_length": pl, "mask_ratio": mr, "mean_mse": mean_mse, "CV": cv, "SEI": sei})
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "Setting & Mean MSE & CV & SEI \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_frontier_metrics.tex", tex)
    values["tab_frontier_metrics"] = records


def gen_tab_wallclock_steps(ms_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 14: Equivalent Optimizer Steps Under Fixed Wall-Clock Budgets"""
    patch_rows = ms_df[ms_df["mask_ratio"] == 0.4].sort_values("patch_length")
    S = 20  # steps per epoch
    lines = []
    for _, r in patch_rows.iterrows():
        pl = int(r["patch_length"])
        mean_time = float(r["epoch_time_mean_sec"])
        step_time = mean_time / S if S > 0 else 1
        s30 = int(30 / step_time)
        s60 = int(60 / step_time)
        s120 = int(120 / step_time)
        lines.append(f"{pl} & {s30} & {s60} & {s120}")
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "Patch $P$ & Steps in 30s & Steps in 60s & Steps in 120s \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_wallclock_steps.tex", tex)


def gen_tab_pairwise_ci(runs_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 15: Pairwise Difference CIs vs Baseline (P=16, rho=0.4)"""
    baseline_runs = runs_df[(runs_df["mask_ratio"] == 0.4) & (runs_df["patch_length"] == 16)]
    b_mse = baseline_runs["masked_mse"].values
    b_time = baseline_runs["epoch_time_sec"].values
    n_b = len(b_mse)

    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}

    configs = [
        (0.2, 16, "$P=16,\\rho=0.2$"),
        (0.6, 16, "$P=16,\\rho=0.6$"),
        (0.4, 8,  "$P=8,\\rho=0.4$"),
        (0.4, 32, "$P=32,\\rho=0.4$"),
    ]

    lines = []
    records = []
    for mr, pl, label in configs:
        comp_runs = runs_df[(runs_df["mask_ratio"] == mr) & (runs_df["patch_length"] == pl)]
        c_mse = comp_runs["masked_mse"].values
        c_time = comp_runs["epoch_time_sec"].values
        n_c = len(c_mse)

        delta_mse = float(np.mean(c_mse) - np.mean(b_mse))
        delta_time = float(np.mean(c_time) - np.mean(b_time))

        # Welch-style SE
        se_mse = math.sqrt(np.var(c_mse, ddof=1) / n_c + np.var(b_mse, ddof=1) / n_b)
        se_time = math.sqrt(np.var(c_time, ddof=1) / n_c + np.var(b_time, ddof=1) / n_b)

        # Conservative df = min(n_a, n_b) - 1
        df = min(n_c, n_b)
        tc = t_crit.get(df, 4.303)

        mse_lo = delta_mse - tc * se_mse
        mse_hi = delta_mse + tc * se_mse
        time_lo = delta_time - tc * se_time
        time_hi = delta_time + tc * se_time

        mse_str = f"${delta_mse:+.4f}\\;[{mse_lo:+.4f},\\;{mse_hi:+.4f}]$"
        time_str = f"${delta_time:+.4f}\\;[{time_lo:+.4f},\\;{time_hi:+.4f}]$"

        label_tex = f"$({label.strip('$')})$ vs baseline"
        lines.append(f"{label} vs baseline & {mse_str} & {time_str}")
        records.append({
            "comparison": f"({mr},{pl}) vs baseline",
            "delta_mse": delta_mse, "mse_ci_lo": mse_lo, "mse_ci_hi": mse_hi,
            "delta_time": delta_time, "time_ci_lo": time_lo, "time_ci_hi": time_hi,
        })

    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "Comparison & Masked MSE Difference (95\\% CI) & Epoch Time Difference in Seconds (95\\% CI) \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_pairwise_ci.tex", tex)
    values["tab_pairwise_ci"] = records


def gen_tab_run_matrix(runs_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 16: Raw Per-Run Metrics"""
    lines = []
    for idx, (_, r) in enumerate(runs_df.iterrows(), 1):
        lines.append(
            f"R{idx} & {int(r['seed'])} & {int(r['data_seed'])} & "
            f"{int(r['patch_length'])} & {r['mask_ratio']:.1f} & "
            f"{_fmt(r['masked_mse'], 6)} & {_fmt(r['epoch_time_sec'])}"
        )
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "Run ID & Seed & Data Seed & $P$ & $\\rho$ & Masked MSE & Epoch Time (s) \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_run_matrix.tex", tex)


def gen_tab_seed_winners(runs_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 17: Seed-Level Winning Configuration"""
    seeds = sorted(runs_df["seed"].unique())
    lines = []
    records = []
    for s in seeds:
        seed_runs = runs_df[runs_df["seed"] == s]
        best_idx = seed_runs["masked_mse"].idxmin()
        best = seed_runs.loc[best_idx]
        pl = int(best["patch_length"])
        mr = float(best["mask_ratio"])
        mse = float(best["masked_mse"])
        config = f"$P={pl},\\rho={mr}$"
        lines.append(f"{int(s)} & {config} & {_fmt(mse)}")
        records.append({"seed": int(s), "patch_length": pl, "mask_ratio": mr, "masked_mse": mse})
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "Seed & Winning Configuration & Masked MSE \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_seed_winners.tex", tex)
    values["tab_seed_winners"] = records


def gen_tab_friedman_patch(runs_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 18: Friedman Rank Analysis for Patch-Length Settings"""
    patch_runs = runs_df[runs_df["mask_ratio"] == 0.4].copy()
    seeds = sorted(patch_runs["seed"].unique())
    patch_lengths = sorted(patch_runs["patch_length"].unique())

    # Compute per-seed ranks (lower MSE = rank 1)
    rank_sums = {pl: 0 for pl in patch_lengths}
    win_counts = {pl: 0 for pl in patch_lengths}
    for s in seeds:
        seed_data = patch_runs[patch_runs["seed"] == s].sort_values("masked_mse")
        for rank, (_, row) in enumerate(seed_data.iterrows(), 1):
            pl = int(row["patch_length"])
            rank_sums[pl] += rank
            if rank == 1:
                win_counts[pl] += 1

    N = len(seeds)
    k = len(patch_lengths)
    # Friedman Q
    Q = (12.0 / (N * k * (k + 1))) * sum(rs ** 2 for rs in rank_sums.values()) - 3 * N * (k + 1)
    W = Q / (N * (k - 1)) if (N * (k - 1)) > 0 else 0

    lines = []
    for pl in patch_lengths:
        rs = rank_sums[pl]
        mr = rs / N
        wc = win_counts[pl]
        lines.append(f"$P={pl}$ & {rs} & {mr:.2f} & {wc}")
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "Configuration & Rank Sum $R_j$ & Mean Rank & Win Count \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_friedman_patch.tex", tex)
    values["friedman"] = {"Q": Q, "W": W, "rank_sums": {str(k): v for k, v in rank_sums.items()}}


def gen_tab_checkpoint_sweep(ckpt_summary: Dict, ckpt_df: "pd.DataFrame", tables_dir: Path, values: Dict):
    """Table 10: Final-Checkpoint Sweep Summary"""
    n_ckpts = len(ckpt_df)
    best = ckpt_df.loc[ckpt_df["masked_mse_eval"].idxmin()]
    final = ckpt_df.iloc[-1]
    drift = (float(final["masked_mse_eval"]) - float(best["masked_mse_eval"])) / float(best["masked_mse_eval"]) * 100

    mean_mse = float(ckpt_df["masked_mse_eval"].mean())
    std_mse = float(ckpt_df["masked_mse_eval"].std())
    mean_tput = float(ckpt_df["tokens_per_sec"].mean())
    std_tput = float(ckpt_df["tokens_per_sec"].std())
    ft_mse_min = float(ckpt_df["full_token_mse"].min())
    ft_mse_max = float(ckpt_df["full_token_mse"].max())

    step_range = f"{ckpt_df['step'].min() / 1e6:.3f}M--{ckpt_df['step'].max() / 1e6:.3f}M"

    lines = [
        f"Checkpoint range & {step_range} steps",
        f"Best masked MSE & {_fmt(float(best['masked_mse_eval']))} @ step {best['step'] / 1e6:.3f}M",
        f"Final masked MSE & {_fmt(float(final['masked_mse_eval']))} @ step {final['step'] / 1e6:.3f}M",
        f"Relative drift (final vs best) & {_pct(drift)}",
        f"Mean masked MSE $\\pm$ std & ${_fmt(mean_mse)} \\pm {_fmt(std_mse)}$",
        f"Mean throughput $\\pm$ std & ${_fmt(mean_tput, 2)} \\pm {_fmt(std_tput, 2)}$ token/s",
        f"Full-token MSE range & {_fmt(ft_mse_min, 3)} to {_fmt(ft_mse_max, 3)}",
    ]
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "Statistic & Value \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_checkpoint_sweep_summary.tex", tex)
    values["checkpoint_sweep"] = {
        "n_checkpoints": n_ckpts,
        "best_masked_mse": float(best["masked_mse_eval"]),
        "best_step": int(best["step"]),
        "final_masked_mse": float(final["masked_mse_eval"]),
        "drift_pct": drift,
        "mean_masked_mse": mean_mse,
        "std_masked_mse": std_mse,
    }


def gen_tab_finetune_transfer(results: Dict, tables_dir: Path, values: Dict):
    """NEW Table: Fine-Tuning Transfer Results (Pretrained vs Scratch)"""
    lines = []
    records = []
    for ds_name, m in results.items():
        pt_mse = m.get("test_mse")
        pt_mae = m.get("test_mae")
        sc_mse = m.get("scratch_test_mse")
        sc_mae = m.get("scratch_test_mae")
        if pt_mse is not None and sc_mse is not None:
            improv = (sc_mse - pt_mse) / sc_mse * 100
        else:
            improv = None
        lines.append(
            f"{ds_name} & {_fmt(pt_mse, 6)} & {_fmt(pt_mae, 6)} & "
            f"{_fmt(sc_mse, 6)} & {_fmt(sc_mae, 6)} & {_pct(improv)}"
        )
        records.append({
            "dataset": ds_name,
            "pretrained_mse": pt_mse, "pretrained_mae": pt_mae,
            "scratch_mse": sc_mse, "scratch_mae": sc_mae,
            "improvement_pct": improv,
        })
    body = " \\\\\n".join(lines)
    tex = (
        "\\toprule\n"
        "Dataset & Pretrained MSE & Pretrained MAE & Scratch MSE & Scratch MAE & Improvement \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_finetune_transfer.tex", tex)
    values["tab_finetune_transfer"] = records


def gen_tab_benchmark(benchmark: Dict, tables_dir: Path, values: Dict):
    """NEW Tables: TSFM vs TimesFM Benchmark Leaderboard + Per-Dataset"""
    datasets_data = benchmark.get("datasets", {})
    agg = benchmark.get("aggregate", {})

    # --- Leaderboard table ---
    if agg:
        lines = []
        sorted_models = sorted(agg.items(), key=lambda x: x[1].get("mean_mse", 1e18))
        for rank, (model_name, stats) in enumerate(sorted_models, 1):
            n_ds = stats.get("n_datasets", 0)
            lines.append(
                f"{rank} & {model_name.replace('_', '\\_')} & "
                f"{_fmt(stats.get('mean_mse'), 2)} & {_fmt(stats.get('mean_mae'), 2)} & {n_ds}"
            )
        body = " \\\\\n".join(lines)
        tex = (
            "\\toprule\n"
            "Rank & Model & Mean MSE & Mean MAE & Datasets \\\\\n"
            "\\midrule\n"
            f"{body} \\\\\n"
            "\\bottomrule\n"
        )
        _write_tex(tables_dir / "tab_benchmark_leaderboard.tex", tex)

    # --- Per-dataset best model table ---
    per_ds_lines = []
    per_ds_records = []
    for ds_name, ds_data in datasets_data.items():
        models = ds_data.get("models", {})
        if not models:
            continue
        best_model = min(models.items(), key=lambda x: x[1].get("mse", 1e18))
        model_name, model_metrics = best_model
        per_ds_lines.append(
            f"{ds_name} & {model_name.replace('_', '\\_')} & "
            f"{_fmt(model_metrics.get('mse'), 4)} & {_fmt(model_metrics.get('mae'), 4)}"
        )
        per_ds_records.append({
            "dataset": ds_name,
            "best_model": model_name,
            "mse": model_metrics.get("mse"),
            "mae": model_metrics.get("mae"),
        })
    body = " \\\\\n".join(per_ds_lines)
    tex = (
        "\\toprule\n"
        "Dataset & Best Model & MSE & MAE \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_benchmark_per_dataset.tex", tex)
    values["tab_benchmark_per_dataset"] = per_ds_records

    # --- TimesFM-only vs best-TSFM comparison table ---
    comparison_lines = []
    comparison_records = []
    for ds_name, ds_data in datasets_data.items():
        models = ds_data.get("models", {})
        timesfm_mse = models.get("timesfm", {}).get("mse")
        timesfm_mae = models.get("timesfm", {}).get("mae")

        # Find best TSFM model for this dataset
        tsfm_models = {k: v for k, v in models.items() if k.startswith("tsfm::")}
        if tsfm_models:
            best_tsfm_name = min(tsfm_models, key=lambda k: tsfm_models[k].get("mse", 1e18))
            best_tsfm_mse = tsfm_models[best_tsfm_name].get("mse")
            best_tsfm_mae = tsfm_models[best_tsfm_name].get("mae")
        else:
            best_tsfm_mse = None
            best_tsfm_mae = None

        winner = "---"
        if timesfm_mse is not None and best_tsfm_mse is not None:
            winner = "TSFM" if best_tsfm_mse < timesfm_mse else "TimesFM"

        comparison_lines.append(
            f"{ds_name} & {_fmt(timesfm_mse, 2)} & {_fmt(best_tsfm_mse, 2)} & \\textbf{{{winner}}}"
        )
        comparison_records.append({
            "dataset": ds_name,
            "timesfm_mse": timesfm_mse,
            "best_tsfm_mse": best_tsfm_mse,
            "winner": winner,
        })

    body = " \\\\\n".join(comparison_lines)
    tex = (
        "\\toprule\n"
        "Dataset & TimesFM MSE & TSFM MSE (best) & Winner \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
    )
    _write_tex(tables_dir / "tab_benchmark_comparison.tex", tex)
    values["tab_benchmark_comparison"] = comparison_records


# ── Figure generators ───────────────────────────────────────────────────────

def gen_fig_benchmark_bar(benchmark: Dict, paper_dir: Path, values: Dict):
    """Bar chart: TimesFM vs best-TSFM MSE per dataset."""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not installed, skipping fig_benchmark_bar")
        return

    datasets_data = benchmark.get("datasets", {})
    ds_names = []
    timesfm_mses = []
    tsfm_mses = []

    for ds_name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Weather"]:
        ds_data = datasets_data.get(ds_name, {})
        models = ds_data.get("models", {})
        tf_mse = models.get("timesfm", {}).get("mse")
        tsfm_models = {k: v for k, v in models.items() if k.startswith("tsfm::")}
        if tsfm_models:
            best_tsfm_mse = min(v.get("mse", 1e18) for v in tsfm_models.values())
        else:
            best_tsfm_mse = None

        if tf_mse is not None and best_tsfm_mse is not None:
            ds_names.append(ds_name)
            timesfm_mses.append(tf_mse)
            tsfm_mses.append(best_tsfm_mse)

    if not ds_names:
        print("  [SKIP] No benchmark data for figure")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ds_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, timesfm_mses, width, label="TimesFM", color="#4285F4", alpha=0.85)
    bars2 = ax.bar(x + width / 2, tsfm_mses, width, label="TSFM (Ours)", color="#EA4335", alpha=0.85)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("TSFM vs TimesFM: Per-Dataset MSE (H=96)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    plt.tight_layout()
    out_path = paper_dir / "fig_benchmark_bar.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def gen_fig_finetune_improvement(results: Dict, paper_dir: Path, values: Dict):
    """Bar chart: Transfer learning improvement per dataset."""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not installed, skipping fig_finetune_improvement")
        return

    ds_names = []
    improvements = []
    for ds_name, m in results.items():
        pt_mse = m.get("test_mse")
        sc_mse = m.get("scratch_test_mse")
        if pt_mse is not None and sc_mse is not None:
            improv = (sc_mse - pt_mse) / sc_mse * 100
            ds_names.append(ds_name.replace("_", "\n"))
            improvements.append(improv)

    if not ds_names:
        print("  [SKIP] No finetune results for figure")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#34A853" if v > 0 else "#EA4335" for v in improvements]
    bars = ax.bar(ds_names, improvements, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("MSE Improvement (%)", fontsize=12)
    ax.set_title("Transfer Learning Benefit: Pretrained vs From-Scratch", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, improvements):
        y_pos = bar.get_height()
        offset = 0.3 if y_pos >= 0 else -0.6
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos + offset,
                f"{val:+.1f}%", ha="center", va="bottom" if y_pos >= 0 else "top", fontsize=10)

    plt.tight_layout()
    out_path = paper_dir / "fig_finetune_improvement.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def gen_fig_checkpoint_trend(ckpt_df: "pd.DataFrame", paper_dir: Path, values: Dict):
    """Dual-axis: full-token MSE + masked MSE over step (regenerated from data)."""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not installed, skipping fig_checkpoint_trend")
        return

    steps = ckpt_df["step"].values
    masked_mse = ckpt_df["masked_mse_eval"].values
    full_mse = ckpt_df["full_token_mse"].values

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_full = "#4285F4"
    color_masked = "#EA4335"

    ax1.set_xlabel("Global Step", fontsize=12)
    ax1.set_ylabel("Full-Token MSE", color=color_full, fontsize=12)
    ax1.plot(steps, full_mse, "o-", color=color_full, label="Full-Token MSE", markersize=4)
    ax1.tick_params(axis="y", labelcolor=color_full)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Masked MSE", color=color_masked, fontsize=12)
    ax2.plot(steps, masked_mse, "s-", color=color_masked, label="Masked MSE", markersize=4)
    ax2.tick_params(axis="y", labelcolor=color_masked)

    fig.suptitle("Late-Checkpoint Objective Trajectory", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = paper_dir / "fig_checkpoint_trend.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def gen_fig_checkpoint_frontier(ckpt_df: "pd.DataFrame", paper_dir: Path, values: Dict):
    """Speed-quality frontier for late checkpoints."""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not installed, skipping fig_checkpoint_frontier")
        return

    masked_mse = ckpt_df["masked_mse_eval"].values
    tput = ckpt_df["tokens_per_sec"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(tput, masked_mse, c=ckpt_df["step"].values,
                         cmap="viridis", s=60, edgecolors="white", linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Global Step", fontsize=11)

    ax.set_xlabel("Token Throughput (tokens/s)", fontsize=12)
    ax.set_ylabel("Masked MSE", fontsize=12)
    ax.set_title("Speed-Quality Frontier (Late Checkpoints)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = paper_dir / "fig_checkpoint_frontier.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate all paper tables and figures")
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"),
                        help="Paper directory (default: paper)")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation (only produce .tex tables)")
    args = parser.parse_args()

    if not HAS_PANDAS:
        print("[ERROR] pandas is required. pip install pandas")
        sys.exit(1)

    paper_dir = args.paper_dir
    tables_dir = paper_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    values: Dict[str, Any] = {}

    print("=" * 60)
    print("GENERATING PAPER ARTIFACTS")
    print("=" * 60)

    # ── Load all data sources ───────────────────────────────────
    print("\n[1/3] Loading data sources...")

    det_df = _safe_read_csv(Path("experiments/ablation_summary_det.csv"))
    ms_df = _safe_read_csv(Path("experiments/multiseed_summary.csv"))
    runs_df = _safe_read_csv(Path("experiments/multiseed_runs.csv"))
    eff_df = _safe_read_csv(Path("experiments/multiseed_effects.csv"))
    ckpt_df = _safe_read_csv(Path("experiments/checkpoint_eval/final_checkpoint_eval.csv"))
    ckpt_summary = _safe_read_json(Path("experiments/checkpoint_eval/final_checkpoint_eval_summary.json"))
    ft_results = _safe_read_json(Path("finetuning_results/results.json"))
    benchmark = _safe_read_json(Path("finetuning_results/benchmark_results.json"))

    loaded_count = sum(1 for x in [det_df, ms_df, runs_df, eff_df, ckpt_df, ckpt_summary, ft_results, benchmark]
                       if x is not None)
    print(f"  Loaded {loaded_count}/8 data sources")

    # ── Generate LaTeX table snippets ───────────────────────────
    print("\n[2/3] Generating LaTeX tables...")

    if det_df is not None:
        gen_tab_mask_ablation(det_df, tables_dir, values)
        gen_tab_patch_ablation(det_df, tables_dir, values)

    if ms_df is not None:
        gen_tab_mask_multiseed(ms_df, tables_dir, values)
        gen_tab_patch_multiseed(ms_df, tables_dir, values)
        gen_tab_frontier_metrics(ms_df, tables_dir, values)
        gen_tab_wallclock_steps(ms_df, tables_dir, values)

    if eff_df is not None:
        gen_tab_effects(eff_df, tables_dir, values)

    if runs_df is not None:
        gen_tab_pairwise_ci(runs_df, tables_dir, values)
        gen_tab_run_matrix(runs_df, tables_dir, values)
        gen_tab_seed_winners(runs_df, tables_dir, values)
        gen_tab_friedman_patch(runs_df, tables_dir, values)

    if ckpt_df is not None and ckpt_summary is not None:
        gen_tab_checkpoint_sweep(ckpt_summary, ckpt_df, tables_dir, values)

    if ft_results is not None:
        gen_tab_finetune_transfer(ft_results, tables_dir, values)

    if benchmark is not None:
        gen_tab_benchmark(benchmark, tables_dir, values)

    # ── Generate figures ────────────────────────────────────────
    if not args.skip_figures:
        print("\n[3/3] Generating figures...")

        if benchmark is not None:
            gen_fig_benchmark_bar(benchmark, paper_dir, values)

        if ft_results is not None:
            gen_fig_finetune_improvement(ft_results, paper_dir, values)

        if ckpt_df is not None:
            gen_fig_checkpoint_trend(ckpt_df, paper_dir, values)
            gen_fig_checkpoint_frontier(ckpt_df, paper_dir, values)
    else:
        print("\n[3/3] Skipping figures (--skip-figures)")

    # ── Save paper_values.json ──────────────────────────────────
    values_path = paper_dir / "paper_values.json"
    with open(values_path, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2, default=str)
    print(f"\n  [OK] {values_path}")

    # ── Summary ─────────────────────────────────────────────────
    tex_files = list(tables_dir.glob("*.tex"))
    fig_files = list(paper_dir.glob("fig_*.png"))
    print(f"\n{'=' * 60}")
    print(f"DONE: {len(tex_files)} LaTeX tables + {len(fig_files)} figures")
    print(f"  Tables → {tables_dir}")
    print(f"  Figures → {paper_dir}")
    print(f"  Values → {values_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
