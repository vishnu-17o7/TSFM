"""
Plot publication-quality training curves from multi-seed ablation metrics JSON files.

Usage:
  python experiments/training_curves.py --metrics-dir experiments/multiseed --output training_curves.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _load_metrics_files(metrics_dir: Path, pattern: str = "*.json") -> List[Dict]:
    metrics_files = sorted(metrics_dir.glob(pattern))
    payloads: List[Dict] = []
    for mf in metrics_files:
        try:
            with open(mf, "r") as f:
                payloads.append(json.load(f))
        except Exception as e:
            print(f"[WARN] Skipping {mf}: {e}")
    return payloads


def _group_payloads(payloads: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    groups: Dict[str, Dict[str, List[Dict]]] = {}
    for p in payloads:
        label = p.get("run_name", "unknown").split("_s")[0].replace("ms_", "")
        seed = p.get("config", {}).get("seed", "?")
        groups.setdefault(label, {})
        groups[label].setdefault("runs", []).append(p)
    return groups


def _extract_epoch_curves(group: Dict) -> Dict[str, np.ndarray]:
    train_curves = []
    val_curves = []
    for run in group.get("runs", []):
        train_losses = []
        val_losses = []
        for ep in run.get("epochs", []):
            t = ep.get("train_masked_mse") or ep.get("masked_mse")
            v = ep.get("val_masked_mse")
            if t is not None:
                train_losses.append(float(t))
            if v is not None:
                val_losses.append(float(v))
        train_curves.append(train_losses)
        val_curves.append(val_losses)

    max_len = max((len(c) for c in train_curves), default=0)
    train_arr = np.full((len(train_curves), max_len), np.nan)
    for i, c in enumerate(train_curves):
        train_arr[i, :len(c)] = c

    max_val = max((len(c) for c in val_curves), default=0)
    val_arr = np.full((len(val_curves), max_val), np.nan) if val_curves else None
    if val_arr is not None:
        for i, c in enumerate(val_curves):
            if c:
                val_arr[i, :len(c)] = c
    return {"train": train_arr, "val": val_arr}


def _plot_curves(
    groups: Dict[str, Dict[str, List[Dict]]],
    output_path: Path,
    figsize: tuple = (14, 5),
    dpi: int = 150,
) -> None:
    if not HAS_MPL:
        print("[ERROR] matplotlib is not installed. pip install matplotlib")
        return

    if not groups:
        print("[WARN] No groups to plot.")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 10)))
    n_configs = len(groups)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    for idx, (label, group) in enumerate(sorted(groups.items())):
        curves = _extract_epoch_curves(group)
        color = colors[idx % len(colors)]

        train = curves["train"]
        if train.size > 0 and not np.all(np.isnan(train)):
            mean_train = np.nanmean(train, axis=0)
            std_train = np.nanstd(train, axis=0)
            epochs = np.arange(1, len(mean_train) + 1)
            axes[0].plot(epochs, mean_train, color=color, label=label, linewidth=1.5)
            if train.shape[0] > 1:
                axes[0].fill_between(epochs, mean_train - std_train, mean_train + std_train,
                                     color=color, alpha=0.15)

        val = curves.get("val")
        if val is not None and val.size > 0 and not np.all(np.isnan(val)):
            mean_val = np.nanmean(val, axis=0)
            std_val = np.nanstd(val, axis=0)
            epochs = np.arange(1, len(mean_val) + 1)
            axes[1].plot(epochs, mean_val, color=color, label=label, linewidth=1.5)
            if val.shape[0] > 1:
                axes[1].fill_between(epochs, mean_val - std_val, mean_val + std_val,
                                     color=color, alpha=0.15)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Masked MSE")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Masked MSE")
    axes[1].set_title("Validation Loss")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"TSFM Ablation Training Curves ({n_configs} configs)", fontsize=13, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Training curves saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from multi-seed ablation metrics.")
    parser.add_argument("--metrics-dir", type=Path, default=Path("experiments/multiseed"),
                        help="Directory containing per-run JSON metrics files.")
    parser.add_argument("--output", type=Path, default=Path("experiments/training_curves.png"),
                        help="Output path for the plot.")
    parser.add_argument("--pattern", type=str, default="*.json",
                        help="Glob pattern for metrics files.")
    parser.add_argument("--figsize", nargs=2, type=float, default=[14, 5],
                        help="Figure size (width height).")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output image DPI.")
    parser.add_argument("--ci-band", action="store_true", default=True,
                        help="Show std bands around mean curves.")

    args = parser.parse_args()

    if not HAS_MPL:
        print("[ERROR] matplotlib is not installed. Install with: pip install matplotlib")
        return

    payloads = _load_metrics_files(args.metrics_dir, args.pattern)
    if not payloads:
        print(f"[ERROR] No JSON metrics files found in {args.metrics_dir}")
        return

    groups = _group_payloads(payloads)
    print(f"[INFO] Loaded {len(payloads)} runs across {len(groups)} configurations")
    for label, group in sorted(groups.items()):
        print(f"  {label}: {len(group.get('runs', []))} seeds")

    _plot_curves(groups, args.output, figsize=tuple(args.figsize), dpi=args.dpi)


if __name__ == "__main__":
    main()
