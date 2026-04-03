import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from train_tsfm import TSFMPretrain, WindowDataset, load_all_series, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TSFM checkpoints on a deterministic validation subset and export plots."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-pattern", type=str, default="checkpoint_epoch_*_step_*.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/checkpoint_eval"))

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-mask-ratio", type=float, default=0.5)

    parser.add_argument("--feature-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic-length", type=int, default=1024)
    parser.add_argument("--progress-every-rows", type=int, default=0)
    parser.add_argument("--feature-workers", type=int, default=1)
    parser.add_argument("--max-rows-per-feature-file", type=int, default=500)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-eval-windows", type=int, default=8192)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--reuse-csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled and output CSV exists, skip checkpoint evaluation and regenerate plots from CSV.",
    )
    return parser.parse_args()


def _choose_device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            out[key[len("_orig_mod.") :]] = value
        else:
            out[key] = value
    return out


def _infer_architecture(state_dict: Dict[str, torch.Tensor], fallback_context_length: int) -> Dict[str, int]:
    cleaned = _strip_compile_prefix(state_dict)

    conv_w = cleaned["patch_embedding.value_embedding.weight"]
    pos = cleaned["patch_embedding.position_embedding"]

    patch_length = int(conv_w.shape[-1])
    embed_dim = int(conv_w.shape[0])
    num_patches = int(pos.shape[1])
    context_length = int(patch_length * num_patches)
    if context_length <= 0:
        context_length = int(fallback_context_length)

    layer_pattern = re.compile(r"transformer\.encoder\.layers\.(\d+)\.")
    layer_ids = []
    for key in cleaned.keys():
        match = layer_pattern.search(key)
        if match:
            layer_ids.append(int(match.group(1)))
    if not layer_ids:
        raise ValueError("Could not infer encoder depth from checkpoint state dict")
    num_layers = max(layer_ids) + 1

    return {
        "context_length": context_length,
        "patch_length": patch_length,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
    }


def _parse_epoch_step(path: Path) -> Tuple[int, int]:
    match = re.search(r"checkpoint_epoch_(\d+)_step_(\d+)\.pt$", path.name)
    if match is None:
        return -1, -1
    return int(match.group(1)), int(match.group(2))


@torch.no_grad()
def _eval_model(
    *,
    model: TSFMPretrain,
    dataloader: DataLoader,
    device: torch.device,
    eval_mask_ratio: float,
) -> Dict[str, float]:
    model.eval()

    masked_losses: List[float] = []
    full_losses: List[float] = []
    total_tokens = 0

    started = time.perf_counter()
    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

        reconstructed, original, mask = model(batch, mask_ratio=eval_mask_ratio)
        if mask.any():
            masked_mse = F.mse_loss(reconstructed[mask], original[mask])
            masked_losses.append(float(masked_mse.item()))

        x_norm = model.revin(batch, mode="norm")
        x_series = x_norm.squeeze(-1)
        embedded = model.patch_embedding(x_series)
        encoded = model.transformer(embedded)
        full_recon = model.reconstruction_head(encoded)
        full_mse = F.mse_loss(full_recon, embedded)
        full_losses.append(float(full_mse.item()))

        batch_size = int(batch.shape[0])
        n_tokens = int(embedded.shape[1])
        total_tokens += batch_size * n_tokens

    elapsed = time.perf_counter() - started
    masked_mean = float(np.mean(masked_losses)) if masked_losses else float("nan")
    full_mean = float(np.mean(full_losses)) if full_losses else float("nan")
    tokens_per_sec = float(total_tokens / max(elapsed, 1e-8))

    return {
        "eval_time_sec": float(elapsed),
        "masked_mse_eval": masked_mean,
        "full_token_mse": full_mean,
        "tokens_per_sec": tokens_per_sec,
        "eval_batches": int(len(masked_losses)),
    }


def _make_loader(args: argparse.Namespace) -> Tuple[DataLoader, Dict[str, object]]:
    series_list, loaded_files, source_mode = load_all_series(
        args.data_dir,
        feature_fallback=args.feature_fallback,
        synthetic_length=args.synthetic_length,
        seed=args.seed,
        progress_every_rows=args.progress_every_rows,
        feature_workers=args.feature_workers,
        max_rows_per_feature_file=args.max_rows_per_feature_file,
    )

    dataset = WindowDataset(series_list, context_length=args.context_length, stride=args.stride)
    total_windows = len(dataset)
    val_size = max(1, int(total_windows * args.val_ratio))
    train_size = total_windows - val_size

    split_gen = torch.Generator()
    split_gen.manual_seed(args.seed)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=split_gen)

    max_eval_windows = args.max_eval_windows
    if max_eval_windows > 0 and len(val_dataset) > max_eval_windows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(val_dataset), size=max_eval_windows, replace=False)
        idx.sort()
        val_dataset = Subset(val_dataset, idx.tolist())

    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    meta: Dict[str, object] = {
        "source_mode": source_mode,
        "loaded_files": int(len(loaded_files)),
        "loaded_series": int(len(series_list)),
        "total_windows": int(total_windows),
        "val_windows": int(len(val_dataset)),
    }
    return loader, meta


def _plot_trend(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    fig.patch.set_facecolor("#0B1020")
    ax.set_facecolor("#121A2F")

    x = df["step"].to_numpy()
    y_full = df["full_token_mse"].to_numpy()
    y_mask = df["masked_mse_eval"].to_numpy()

    line_full = ax.plot(
        x,
        y_full,
        color="#5EEAD4",
        linewidth=2.8,
        marker="o",
        markersize=4.5,
        label="Full-token MSE",
    )[0]

    ax2 = ax.twinx()
    line_mask = ax2.plot(
        x,
        y_mask,
        color="#F59E0B",
        linewidth=2.4,
        marker="D",
        markersize=3.8,
        label="Masked MSE (rho=0.5)",
    )[0]

    best_idx = int(np.nanargmin(y_full))
    ax.scatter([x[best_idx]], [y_full[best_idx]], s=120, color="#22D3EE", edgecolors="#FFFFFF", linewidths=1.0, zorder=5)
    ax.annotate(
        f"best @ step {int(x[best_idx]):,}\nMSE={y_full[best_idx]:.4f}",
        xy=(x[best_idx], y_full[best_idx]),
        xytext=(10, -40),
        textcoords="offset points",
        color="#E2E8F0",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#CBD5E1", lw=1.0),
    )

    ax.grid(True, color="#334155", alpha=0.35, linewidth=0.8)
    ax.tick_params(colors="#E2E8F0")
    ax2.tick_params(colors="#FCD34D")
    ax.set_title("Final Checkpoint Sweep: Objective Trajectory", color="#F8FAFC", fontsize=13, weight="bold")
    ax.set_xlabel("Global step", color="#E2E8F0")
    ax.set_ylabel("Full-token MSE", color="#5EEAD4")
    ax2.set_ylabel("Masked MSE", color="#F59E0B")

    for spine in ax.spines.values():
        spine.set_color("#64748B")

    for spine in ax2.spines.values():
        spine.set_color("#64748B")

    leg = ax.legend(
        handles=[line_full, line_mask],
        frameon=True,
        facecolor="#0F172A",
        edgecolor="#334155",
        fontsize=8,
        loc="upper left",
    )
    for text in leg.get_texts():
        text.set_color("#E2E8F0")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_quality_speed(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    fig.patch.set_facecolor("#FFF8EE")
    ax.set_facecolor("#FFFCF5")

    sc = ax.scatter(
        df["tokens_per_sec"],
        df["masked_mse_eval"],
        c=df["step"],
        cmap="viridis",
        s=90,
        alpha=0.95,
        edgecolors="#0F172A",
        linewidths=0.6,
    )
    ax.plot(df["tokens_per_sec"], df["masked_mse_eval"], color="#0F766E", alpha=0.55, linewidth=1.5)

    best_row = df.loc[df["masked_mse_eval"].idxmin()]
    ax.scatter(
        [best_row["tokens_per_sec"]],
        [best_row["masked_mse_eval"]],
        s=170,
        marker="*",
        color="#DC2626",
        edgecolors="#7F1D1D",
        linewidths=0.7,
        zorder=6,
        label="Best quality checkpoint",
    )

    ax.set_title("Speed-Quality Frontier Across Final Checkpoints", fontsize=13, weight="bold", color="#0F172A")
    ax.set_xlabel("Tokens per second", color="#0F172A")
    ax.set_ylabel("Masked MSE (lower is better)", color="#0F172A")
    ax.grid(True, linestyle="--", alpha=0.35)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Global step", color="#0F172A")
    cbar.ax.yaxis.set_tick_params(color="#0F172A")
    plt.setp(cbar.ax.get_yticklabels(), color="#0F172A")

    ax.legend(frameon=True, facecolor="#FEF3C7", edgecolor="#F59E0B")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _checkpoint_paths(checkpoints_dir: Path, pattern: str) -> Iterable[Path]:
    paths = sorted(checkpoints_dir.glob(pattern))
    paths.sort(key=lambda p: _parse_epoch_step(p)[1])
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    csv_out = args.output_dir / "final_checkpoint_eval.csv"
    json_out = args.output_dir / "final_checkpoint_eval_summary.json"
    fig_trend = args.output_dir / "checkpoint_mse_trend.png"
    fig_frontier = args.output_dir / "checkpoint_speed_quality.png"

    if args.reuse_csv and csv_out.exists():
        df = pd.read_csv(csv_out)
        _plot_trend(df, fig_trend)
        _plot_quality_speed(df, fig_frontier)
        print(f"Reused CSV: {csv_out}")
        print(f"Saved: {fig_trend}")
        print(f"Saved: {fig_frontier}")
        return

    device = _choose_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    loader, data_meta = _make_loader(args)
    ckpts = list(_checkpoint_paths(args.checkpoints_dir, args.checkpoint_pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoint_pattern} in {args.checkpoints_dir}")

    print(f"Device: {device}")
    print(f"Validation windows: {data_meta['val_windows']}")
    print(f"Checkpoints to evaluate: {len(ckpts)}")

    rows: List[Dict[str, object]] = []
    for ckpt_path in ckpts:
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model"]
        arch = _infer_architecture(state_dict, fallback_context_length=args.context_length)

        model = TSFMPretrain(
            context_length=arch["context_length"],
            patch_length=arch["patch_length"],
            embed_dim=arch["embed_dim"],
            num_heads=args.num_heads,
            num_layers=arch["num_layers"],
            dropout=args.dropout,
        ).to(device)
        model.load_state_dict(_strip_compile_prefix(state_dict), strict=True)

        epoch, step = _parse_epoch_step(ckpt_path)
        metrics = _eval_model(
            model=model,
            dataloader=loader,
            device=device,
            eval_mask_ratio=args.eval_mask_ratio,
        )

        row: Dict[str, object] = {
            "checkpoint": ckpt_path.name,
            "epoch": int(checkpoint.get("epoch", epoch)),
            "step": int(checkpoint.get("step", step)),
            "patch_length": int(arch["patch_length"]),
            "context_length": int(arch["context_length"]),
            "embed_dim": int(arch["embed_dim"]),
            "num_layers": int(arch["num_layers"]),
            **metrics,
        }
        rows.append(row)
        print(
            f"[{row['checkpoint']}] step={row['step']:,} "
            f"full_mse={row['full_token_mse']:.6f} masked_mse={row['masked_mse_eval']:.6f}"
        )

    df = pd.DataFrame(rows).sort_values(by="step").reset_index(drop=True)

    best_full = df.loc[df["full_token_mse"].idxmin()].to_dict()
    best_masked = df.loc[df["masked_mse_eval"].idxmin()].to_dict()

    df.to_csv(csv_out, index=False)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "eval_mask_ratio": float(args.eval_mask_ratio),
        "data": data_meta,
        "architecture": {
            "num_heads_eval": int(args.num_heads),
            "dropout_eval": float(args.dropout),
        },
        "best_full_token_mse": best_full,
        "best_masked_mse_eval": best_masked,
    }
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_trend(df, fig_trend)
    _plot_quality_speed(df, fig_frontier)

    print(f"Saved: {csv_out}")
    print(f"Saved: {json_out}")
    print(f"Saved: {fig_trend}")
    print(f"Saved: {fig_frontier}")


if __name__ == "__main__":
    main()