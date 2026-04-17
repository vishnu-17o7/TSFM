import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from train_tsfm import TSFMPretrain, WindowDataset, _as_jsonable, _write_metrics, load_all_series, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic multi-seed TSFM ablations with shared data loading.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/multiseed"))
    parser.add_argument("--summary-csv", type=Path, default=Path("experiments/multiseed_summary.csv"))
    parser.add_argument("--seeds", type=str, default="11,42,123")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=20)
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


def _build_ablation_grid() -> List[Dict[str, float]]:
    return [
        {"label": "mask_0p2", "mask_ratio": 0.2, "patch_length": 16},
        {"label": "mask_0p4", "mask_ratio": 0.4, "patch_length": 16},
        {"label": "mask_0p6", "mask_ratio": 0.6, "patch_length": 16},
        {"label": "patch_8", "mask_ratio": 0.4, "patch_length": 8},
        {"label": "patch_32", "mask_ratio": 0.4, "patch_length": 32},
    ]


def _run_single(
    *,
    cfg: Dict[str, float],
    seed: int,
    data_seed: int,
    args: argparse.Namespace,
    dataset: WindowDataset,
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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)
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
    dataloader = DataLoader(dataset, **dl_kwargs)

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
                "synthetic_length": args.synthetic_length,
                "progress_every_rows": args.progress_every_rows,
                "feature_workers": args.feature_workers,
                "max_rows_per_feature_file": args.max_rows_per_feature_file,
                "mask_ratio": cfg["mask_ratio"],
                "patch_length": int(cfg["patch_length"]),
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
            "total_windows": len(dataset),
        },
        "model": {
            "total_params": int(sum(p.numel() for p in model.parameters())),
        },
        "epochs": [],
        "summary": {},
    }

    accum_steps = max(1, int(args.gradient_accumulation_steps))
    total_batches = len(dataloader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        step_count = 0
        micro_step = 0

        max_steps = total_batches
        if args.max_steps_per_epoch > 0:
            max_steps = min(args.max_steps_per_epoch, total_batches)

        for batch in dataloader:
            if args.max_steps_per_epoch > 0 and step_count >= args.max_steps_per_epoch:
                break

            batch = batch.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                reconstructed, original, mask = model(batch, mask_ratio=float(cfg["mask_ratio"]))
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

            running_loss += loss.item() * accum_steps
            step_count += 1

        if micro_step % accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = running_loss / max(step_count, 1)
        epoch_time_sec = time.perf_counter() - epoch_start
        metrics_payload["epochs"].append(
            {
                "epoch": epoch,
                "steps": int(step_count),
                "masked_mse": float(avg_loss),
                "epoch_time_sec": float(epoch_time_sec),
            }
        )

    losses = [ep["masked_mse"] for ep in metrics_payload["epochs"]]
    final_loss = float(losses[-1]) if losses else None
    best_loss = float(min(losses)) if losses else None
    elapsed_sec = time.perf_counter() - run_start

    metrics_payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
    metrics_payload["summary"] = {
        "total_train_time_sec": float(elapsed_sec),
        "best_masked_mse": best_loss,
        "final_masked_mse": final_loss,
        "checkpoint_path": None,
    }

    return metrics_payload


def _summary_rows(run_payloads: List[Dict[str, object]], batch_size: int, context_length: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    t_critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }

    flat: List[Dict[str, object]] = []
    for payload in run_payloads:
        cfg = payload["config"]
        epoch = payload["epochs"][-1]
        patch = int(cfg["patch_length"])
        steps = int(epoch["steps"])
        epoch_time = float(epoch["epoch_time_sec"])
        token_count = context_length // patch
        token_throughput = (steps * batch_size * token_count) / max(epoch_time, 1e-8)
        flat.append(
            {
                "label": payload["run_name"].split("_s")[0].replace("ms_", ""),
                "seed": int(cfg["seed"]),
                "mask_ratio": float(cfg["mask_ratio"]),
                "patch_length": patch,
                "steps": steps,
                "masked_mse": float(epoch["masked_mse"]),
                "epoch_time_sec": epoch_time,
                "token_throughput": token_throughput,
            }
        )

    df = pd.DataFrame(flat)
    group_cols = ["label", "mask_ratio", "patch_length"]
    grouped = df.groupby(group_cols, as_index=False)

    for _, g in grouped:
        n = int(len(g))
        mse_mean = float(g["masked_mse"].mean())
        mse_std = float(g["masked_mse"].std(ddof=1)) if n > 1 else 0.0
        sem = mse_std / math.sqrt(n) if n > 1 else 0.0
        t = t_critical.get(n - 1, 1.96) if n > 1 else 0.0
        ci95 = t * sem

        time_mean = float(g["epoch_time_sec"].mean())
        thr_mean = float(g["token_throughput"].mean())

        rows.append(
            {
                "label": g["label"].iloc[0],
                "mask_ratio": float(g["mask_ratio"].iloc[0]),
                "patch_length": int(g["patch_length"].iloc[0]),
                "n_seeds": n,
                "steps_mean": float(g["steps"].mean()),
                "masked_mse_mean": mse_mean,
                "masked_mse_std": mse_std,
                "masked_mse_ci95": ci95,
                "epoch_time_mean_sec": time_mean,
                "token_throughput_mean": thr_mean,
            }
        )

    out = pd.DataFrame(rows).sort_values(by=["patch_length", "mask_ratio"]).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    seeds = _parse_seed_list(args.seeds)
    ablations = _build_ablation_grid()

    cpu_count = os.cpu_count() or 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    print(f"Device: {device}")
    print(f"CPU cores: {cpu_count}")
    print(f"Seeds: {seeds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_payloads: List[Dict[str, object]] = []

    for seed in seeds:
        print(f"[DATA] Loading corpus with seed={seed}")
        series_list, loaded_files, source_mode = load_all_series(
            args.data_dir,
            feature_fallback=args.feature_fallback,
            synthetic_length=args.synthetic_length,
            seed=seed,
            progress_every_rows=args.progress_every_rows,
            feature_workers=args.feature_workers,
            max_rows_per_feature_file=args.max_rows_per_feature_file,
        )
        dataset = WindowDataset(series_list, context_length=args.context_length, stride=args.stride)
        data_meta = {
            "source_mode": source_mode,
            "loaded_files": len(loaded_files),
            "loaded_series": len(series_list),
        }

        for cfg in ablations:
            print(f"[RUN] {cfg['label']} seed={seed}")
            payload = _run_single(
                cfg=cfg,
                seed=seed,
                data_seed=seed,
                args=args,
                dataset=dataset,
                data_meta=data_meta,
                cpu_count=cpu_count,
                device=device,
                use_amp=use_amp,
            )
            run_payloads.append(payload)

            if args.save_json:
                out_path = args.output_dir / f"{cfg['label']}_seed{seed}.json"
                _write_metrics(out_path, payload)

            final_mse = payload["summary"]["final_masked_mse"]
            elapsed = payload["summary"]["total_train_time_sec"]
            print(f"[DONE] {cfg['label']} seed={seed} | mse={final_mse:.6f} | time={elapsed:.2f}s")

    summary_df = _summary_rows(
        run_payloads,
        batch_size=args.batch_size,
        context_length=args.context_length,
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_csv, index=False)
    print(f"Saved summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
