"""
Evaluation and visualization utilities for fine-tuning results.
Generates comparison plots and performance metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Try to import optional plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class ResultsAnalyzer:
    """Analyze and visualize fine-tuning results."""

    @staticmethod
    def load_results(results_file: Path) -> Dict:
        """Load results from JSON file."""
        with open(results_file, "r") as f:
            return json.load(f)

    @staticmethod
    def print_summary(results: Dict) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("FINE-TUNING RESULTS SUMMARY")
        print("=" * 80)

        for dataset_name, metrics in results.items():
            print(f"\n{dataset_name}:")
            print("-" * 40)

            test_mse = metrics.get("test_mse")
            test_mae = metrics.get("test_mae")
            scratch_mse = metrics.get("scratch_test_mse")
            scratch_mae = metrics.get("scratch_test_mae")

            if test_mse is not None:
                print(f"  Pretrained:")
                print(f"    Test MSE:  {test_mse:.6f}")
                print(f"    Test MAE:  {test_mae:.6f}")

            if scratch_mse is not None:
                print(f"  From-Scratch (Baseline):")
                print(f"    Test MSE:  {scratch_mse:.6f}")
                print(f"    Test MAE:  {scratch_mae:.6f}")

                # Calculate improvement
                mse_improvement = (scratch_mse - test_mse) / scratch_mse * 100
                mae_improvement = (scratch_mae - test_mae) / scratch_mae * 100

                print(f"  Improvement (Pretrained vs Scratch):")
                print(f"    MSE: {mse_improvement:+.2f}%")
                print(f"    MAE: {mae_improvement:+.2f}%")

    @staticmethod
    def create_comparison_table(results: Dict) -> pd.DataFrame:
        """Create comparison table."""
        data = []
        for dataset_name, metrics in results.items():
            row = {
                "Dataset": dataset_name,
                "Pretrained MSE": metrics.get("test_mse"),
                "Pretrained MAE": metrics.get("test_mae"),
                "Scratch MSE": metrics.get("scratch_test_mse"),
                "Scratch MAE": metrics.get("scratch_test_mae"),
            }

            # Calculate improvements
            if metrics.get("scratch_test_mse") is not None:
                mse_improvement = (
                    (metrics.get("scratch_test_mse") - metrics.get("test_mse"))
                    / metrics.get("scratch_test_mse")
                    * 100
                )
                mae_improvement = (
                    (metrics.get("scratch_test_mae") - metrics.get("test_mae"))
                    / metrics.get("scratch_test_mae")
                    * 100
                )
                row["MSE Improvement %"] = mse_improvement
                row["MAE Improvement %"] = mae_improvement

            data.append(row)

        return pd.DataFrame(data)

    @staticmethod
    def plot_results(
        results_file: Path,
        output_dir: Path,
        figsize: tuple = (14, 10),
    ) -> None:
        """Plot comparison results."""
        if not HAS_MATPLOTLIB:
            print("[WARN] matplotlib not installed. Skipping plots.")
            return

        results = ResultsAnalyzer.load_results(results_file)
        df = ResultsAnalyzer.create_comparison_table(results)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Bar chart: MSE/MAE comparison
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        datasets = df["Dataset"].tolist()
        x_pos = np.arange(len(datasets))
        width = 0.35

        # MSE comparison
        if "Pretrained MSE" in df.columns and "Scratch MSE" in df.columns:
            axes[0].bar(x_pos - width / 2, df["Pretrained MSE"], width, label="Pretrained", alpha=0.8)
            axes[0].bar(x_pos + width / 2, df["Scratch MSE"], width, label="From-Scratch", alpha=0.8)
            axes[0].set_xlabel("Dataset")
            axes[0].set_ylabel("MSE")
            axes[0].set_title("Test MSE Comparison")
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(datasets)
            axes[0].legend()
            axes[0].grid(axis="y", alpha=0.3)

        # MAE comparison
        if "Pretrained MAE" in df.columns and "Scratch MAE" in df.columns:
            axes[1].bar(x_pos - width / 2, df["Pretrained MAE"], width, label="Pretrained", alpha=0.8)
            axes[1].bar(x_pos + width / 2, df["Scratch MAE"], width, label="From-Scratch", alpha=0.8)
            axes[1].set_xlabel("Dataset")
            axes[1].set_ylabel("MAE")
            axes[1].set_title("Test MAE Comparison")
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(datasets)
            axes[1].legend()
            axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / "comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved comparison plot to {output_path}")
        plt.close()

        # 2. Improvement percentages
        if "MSE Improvement %" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            colors = ["green" if x > 0 else "red" for x in df["MSE Improvement %"]]
            ax.bar(datasets, df["MSE Improvement %"], color=colors, alpha=0.7)
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax.set_ylabel("Improvement (%)")
            ax.set_title("Pre-training Benefit: MSE Improvement over From-Scratch")
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            output_path = output_dir / "improvement.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Saved improvement plot to {output_path}")
            plt.close()

    @staticmethod
    def generate_report(
        results_file: Path,
        output_dir: Path,
    ) -> None:
        """Generate a comprehensive report."""
        results = ResultsAnalyzer.load_results(results_file)
        df = ResultsAnalyzer.create_comparison_table(results)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate markdown report
        report_path = output_dir / "report.md"
        with open(report_path, "w") as f:
            f.write("# Fine-Tuning Results Report\n\n")

            f.write("## Summary Statistics\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")

            f.write("## Key Findings\n\n")

            # Calculate average improvements
            if "MSE Improvement %" in df.columns:
                avg_mse_improvement = df["MSE Improvement %"].mean()
                f.write(f"- **Average MSE Improvement**: {avg_mse_improvement:.2f}%\n")

            if "MAE Improvement %" in df.columns:
                avg_mae_improvement = df["MAE Improvement %"].mean()
                f.write(f"- **Average MAE Improvement**: {avg_mae_improvement:.2f}%\n")

            f.write("\n## Dataset-Specific Results\n\n")
            for dataset_name, metrics in results.items():
                f.write(f"### {dataset_name}\n\n")

                test_mse = metrics.get("test_mse")
                test_mae = metrics.get("test_mae")
                scratch_mse = metrics.get("scratch_test_mse")
                scratch_mae = metrics.get("scratch_test_mae")

                if test_mse is not None:
                    f.write(f"**Pretrained Model:**\n")
                    f.write(f"- Test MSE: {test_mse:.6f}\n")
                    f.write(f"- Test MAE: {test_mae:.6f}\n\n")

                if scratch_mse is not None:
                    f.write(f"**From-Scratch Baseline:**\n")
                    f.write(f"- Test MSE: {scratch_mse:.6f}\n")
                    f.write(f"- Test MAE: {scratch_mae:.6f}\n\n")

                    mse_improvement = (scratch_mse - test_mse) / scratch_mse * 100
                    mae_improvement = (scratch_mae - test_mae) / scratch_mae * 100

                    f.write(f"**Improvement:**\n")
                    f.write(f"- MSE: {mse_improvement:+.2f}%\n")
                    f.write(f"- MAE: {mae_improvement:+.2f}%\n\n")

            f.write("## Methodology\n\n")
            f.write("1. **Linear Probe**: Frozen pre-trained encoder, train only forecasting head\n")
            f.write("2. **Full Fine-tuning**: Unfreeze all layers with lower learning rate (5e-5)\n")
            f.write("3. **Baseline**: Identical architecture trained from scratch\n")
            f.write("4. **Evaluation**: Test MSE and MAE on 20% holdout test set\n\n")

            f.write("## Conclusion\n\n")
            avg_improvement = 0
            if "MSE Improvement %" in df.columns:
                avg_improvement = df["MSE Improvement %"].mean()

            if avg_improvement > 0:
                f.write(
                    f"Pre-training on synthetic time-series data provides consistent benefits:\n"
                    f"On average, pre-trained models achieve **{avg_improvement:.2f}% better MSE** than "
                    f"models trained from scratch.\n"
                    f"This validates the effectiveness of pre-training for learning universal "
                    f"time-series representations.\n"
                )
            elif avg_improvement < 0:
                f.write(
                    f"Pre-trained models show mixed results, with an average "
                    f"**{abs(avg_improvement):.2f}% degradation** vs from-scratch models.\n"
                    f"This may indicate that the pre-training task (masked reconstruction) "
                    f"is not well-aligned with the downstream forecasting task.\n"
                )
            else:
                f.write("Pre-trained and from-scratch models show comparable performance.\n")

        print(f"[INFO] Report saved to {report_path}")

        # Also print to console
        ResultsAnalyzer.print_summary(results)
        print("\n")
        print(df.to_string(index=False))


def main():
    """Analyze results."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze fine-tuning results")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("finetuning_results/results.json"),
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("finetuning_results"),
        help="Output directory for plots and reports",
    )
    parser.add_argument("--plot", action="store_true", default=True, help="Generate plots")
    parser.add_argument("--report", action="store_true", default=True, help="Generate report")

    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"[ERROR] Results file not found: {args.results_file}")
        return

    if args.plot:
        ResultsAnalyzer.plot_results(args.results_file, args.output_dir)

    if args.report:
        ResultsAnalyzer.generate_report(args.results_file, args.output_dir)


if __name__ == "__main__":
    main()
