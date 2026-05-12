"""
Statistical power planning tool for multi-seed TS foundation model experiments.

Given a pilot run CSV (from run_multiseed_ablation.py --estimate-seeds), prints
recommended seed counts to achieve a target CI half-width.

Usage:
  python experiments/seed_count_planner.py --pilot-csv experiments/multiseed_summary.csv
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import pandas as pd


def _estimate_required_seeds(
    pilot_df: pd.DataFrame,
    target_half_width_ratio: float = 0.10,
    use_scipy: bool = True,
) -> pd.DataFrame:
    try:
        from scipy import stats as scipy_stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        if use_scipy:
            print("[WARN] scipy not installed. Falling back to z=1.96 approximation.")
            use_scipy = False

    rows = []
    required_cols = ["label", "mask_ratio", "patch_length", "n_seeds", "train_mse_mean", "train_mse_std"]
    available = [c for c in required_cols if c in pilot_df.columns]

    if "train_mse_mean" not in available or "train_mse_std" not in available:
        raise ValueError("Pilot CSV must contain 'train_mse_mean' and 'train_mse_std' columns. "
                         "Run with --estimate-seeds flag in run_multiseed_ablation.py first.")

    for _, row in pilot_df.iterrows():
        n_pilot = int(row.get("n_seeds", 1))
        mean_val = float(row["train_mse_mean"])
        std_val = float(row["train_mse_std"])

        if std_val <= 0 or n_pilot < 2:
            n_required = n_pilot
        else:
            target_hw = target_half_width_ratio * abs(mean_val) if abs(mean_val) > 1e-12 else target_half_width_ratio
            if target_hw <= 0:
                n_required = n_pilot
            else:
                df_val = n_pilot - 1
                if use_scipy and HAS_SCIPY:
                    t_crit = scipy_stats.t.ppf(0.975, df=df_val)
                else:
                    t_crit = 1.96
                n_required = int(math.ceil((t_crit * std_val / target_hw) ** 2))
                n_required = max(n_pilot, n_required)

        row_data = {
            "label": str(row.get("label", "")),
            "mask_ratio": float(row.get("mask_ratio", 0)),
            "patch_length": int(row.get("patch_length", 0)),
            "n_pilot": n_pilot,
            "mse_mean": mean_val,
            "mse_std": std_val,
            "target_half_width_ratio": target_half_width_ratio,
            "n_required": n_required,
            "seeds_still_needed": max(0, n_required - n_pilot),
        }
        rows.append(row_data)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate required seed count for target statistical precision."
    )
    parser.add_argument("--pilot-csv", type=Path, default=Path("experiments/multiseed_summary.csv"),
                        help="Pilot run summary CSV from run_multiseed_ablation.py.")
    parser.add_argument("--target-half-width-ratio", type=float, default=0.10,
                        help="Target CI half-width as fraction of mean (default: 0.10 = 10%%).")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Save seed count plan to CSV.")

    args = parser.parse_args()

    if not args.pilot_csv.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {args.pilot_csv}")

    pilot_df = pd.read_csv(args.pilot_csv)
    plan_df = _estimate_required_seeds(pilot_df, args.target_half_width_ratio)

    print("\nSeed Count Planning Report")
    print(f"  Target CI half-width: {args.target_half_width_ratio * 100:.0f}% of mean")
    print(f"  Pilot CSV: {args.pilot_csv}")
    print()
    print(plan_df.to_string(index=False))
    print()

    for _, row in plan_df.iterrows():
        needed = int(row["seeds_still_needed"])
        if needed > 0:
            print(f"  {row['label']}: need {needed} more seeds (currently {row['n_pilot']}, target {row['n_required']})")
        else:
            print(f"  {row['label']}: sufficient ({row['n_pilot']} seeds)")

    if args.output_csv is not None:
        plan_df.to_csv(args.output_csv, index=False)
        print(f"\n[INFO] Plan saved to {args.output_csv}")


if __name__ == "__main__":
    main()
