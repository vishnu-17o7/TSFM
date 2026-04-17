"""Print a clean ranked leaderboard from benchmark_results.json."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _format_float(x: float) -> str:
    return f"{x:.6f}"


def _build_rows(aggregate: Dict[str, Any], include_timesfm: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name, stats in aggregate.items():
        is_timesfm = model_name == "timesfm"
        if is_timesfm and not include_timesfm:
            continue

        rows.append(
            {
                "model": model_name,
                "mean_mse": float(stats.get("mean_mse", float("inf"))),
                "mean_mae": float(stats.get("mean_mae", float("inf"))),
                "num_datasets": int(stats.get("num_datasets", 0)),
                "is_timesfm": is_timesfm,
            }
        )

    return rows


def _print_table(rows: List[Dict[str, Any]], sort_by: str, top_k: int) -> None:
    if not rows:
        print("No leaderboard rows found.")
        return

    rows = sorted(rows, key=lambda r: (r[sort_by], r["mean_mse"], r["mean_mae"], r["model"]))
    if top_k > 0:
        rows = rows[:top_k]

    headers = ["Rank", "Model", "Mean MSE", "Mean MAE", "Datasets"]
    widths = [
        len(headers[0]),
        max(len(headers[1]), max(len(r["model"]) for r in rows)),
        len(headers[2]),
        len(headers[3]),
        len(headers[4]),
    ]

    sep = f"+-{'-' * widths[0]}-+-{'-' * widths[1]}-+-{'-' * widths[2]}-+-{'-' * widths[3]}-+-{'-' * widths[4]}-+"
    print(sep)
    print(
        f"| {headers[0].ljust(widths[0])} "
        f"| {headers[1].ljust(widths[1])} "
        f"| {headers[2].rjust(widths[2])} "
        f"| {headers[3].rjust(widths[3])} "
        f"| {headers[4].rjust(widths[4])} |"
    )
    print(sep)

    for idx, row in enumerate(rows, start=1):
        print(
            f"| {str(idx).ljust(widths[0])} "
            f"| {row['model'].ljust(widths[1])} "
            f"| {_format_float(row['mean_mse']).rjust(widths[2])} "
            f"| {_format_float(row['mean_mae']).rjust(widths[3])} "
            f"| {str(row['num_datasets']).rjust(widths[4])} |"
        )

    print(sep)


def _print_dataset_winners(payload: Dict[str, Any], include_timesfm: bool) -> None:
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        return

    print("\nBest model per dataset (by MSE):")
    for dataset_name, dataset_info in datasets.items():
        models = dataset_info.get("models", {})
        if not isinstance(models, dict) or not models:
            continue

        filtered = []
        for model_name, stats in models.items():
            if (model_name == "timesfm") and (not include_timesfm):
                continue
            mse = stats.get("mse")
            mae = stats.get("mae")
            if mse is None or mae is None:
                continue
            filtered.append((model_name, float(mse), float(mae)))

        if not filtered:
            continue

        best = min(filtered, key=lambda x: (x[1], x[2], x[0]))
        print(
            f"- {dataset_name}: {best[0]} "
            f"(mse={_format_float(best[1])}, mae={_format_float(best[2])})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print ranked leaderboard from benchmark_results.json")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("finetuning_results/benchmark_results.json"),
        help="Path to benchmark results JSON",
    )
    parser.add_argument(
        "--sort-by",
        choices=["mean_mse", "mean_mae"],
        default="mean_mse",
        help="Sort metric for ranking",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Show only top-k rows (0 means all)",
    )
    parser.add_argument(
        "--exclude-timesfm",
        action="store_true",
        help="Exclude TimesFM baseline from leaderboard",
    )
    parser.add_argument(
        "--show-dataset-winners",
        action="store_true",
        help="Also print per-dataset best model by MSE",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Results file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    aggregate = payload.get("aggregate", {})
    if not isinstance(aggregate, dict) or not aggregate:
        raise ValueError("Input JSON does not contain aggregate metrics.")

    include_timesfm = not args.exclude_timesfm
    rows = _build_rows(aggregate=aggregate, include_timesfm=include_timesfm)

    print("\nBenchmark Leaderboard")
    print(f"Source: {args.input}")
    print(f"Sorted by: {args.sort_by}")
    _print_table(rows=rows, sort_by=args.sort_by, top_k=args.top_k)

    if args.show_dataset_winners:
        _print_dataset_winners(payload=payload, include_timesfm=include_timesfm)


if __name__ == "__main__":
    main()
