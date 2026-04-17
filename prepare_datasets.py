"""Dataset preparation utility for TSFM training and benchmark evaluation."""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Try to import optional dependency.
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class DataDownloader:
    """Download and prepare real time-series datasets."""

    DATASET_URLS: Dict[str, List[str]] = {
        "metro_volume": [
            "https://raw.githubusercontent.com/OreStein/MultiScale/master/dataset/metro_volume.csv",
        ],
        "ETTh1": [
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        ],
        "ETTh2": [
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        ],
        "ETTm1": [
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        ],
        "ETTm2": [
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
        ],
        # Electricity / Traffic / Weather are kept as manual-placement targets because
        # reliable direct raw-csv URLs can vary across mirrors.
        "Electricity": [],
        "Traffic": [],
        "Weather": [],
    }

    BENCHMARK_OUTPUTS: Dict[str, str] = {
        "ETTh1": "ETTh1.csv",
        "ETTh2": "ETTh2.csv",
        "ETTm1": "ETTm1.csv",
        "ETTm2": "ETTm2.csv",
        "Electricity": "electricity.csv",
        "Traffic": "traffic.csv",
        "Weather": "weather.csv",
    }

    TRAFFIC_METRO_SOURCE_CANDIDATES: List[Path] = [
        Path("metro+interstate+traffic+volume") / "Metro_Interstate_Traffic_Volume.csv",
        Path("metro_volume.csv"),
    ]

    @staticmethod
    def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
        """Download file from URL."""
        if not HAS_REQUESTS:
            print("[ERROR] requests library not installed. Install with: pip install requests")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[INFO] Downloading from {url}...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

            print(f"[INFO] Downloaded to {output_path}")
            return True
        except Exception as e:
            print(f"[WARN] Failed download from {url}: {e}")
            return False

    @staticmethod
    def _download_from_candidates(urls: List[str], output_path: Path) -> bool:
        for url in urls:
            if DataDownloader.download_file(url, output_path):
                return True
        return False

    @staticmethod
    def _verify_csv(path: Path) -> bool:
        try:
            df = pd.read_csv(path, nrows=10)
            print(f"[INFO] Verified {path.name}: columns={len(df.columns)}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}")
            return False

    @staticmethod
    def _prepare_traffic_from_metro(data_dir: Path, output_path: Path) -> Optional[Path]:
        """Build traffic.csv from Metro Interstate Traffic Volume CSV when available."""
        for rel_path in DataDownloader.TRAFFIC_METRO_SOURCE_CANDIDATES:
            source = data_dir / rel_path
            if not source.exists():
                continue

            try:
                metro_df = pd.read_csv(source)
            except Exception as e:
                print(f"[WARN] Failed to read metro source {source}: {e}")
                continue

            if "traffic_volume" in metro_df.columns:
                out_df = pd.DataFrame({"traffic_volume": metro_df["traffic_volume"]})
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out_df.to_csv(output_path, index=False)
                print(f"[INFO] Built benchmark Traffic dataset at {output_path} from {source}")
                if DataDownloader._verify_csv(output_path):
                    return output_path
                return None

            print(f"[WARN] Metro source {source} does not contain 'traffic_volume' column.")

        return None

    @staticmethod
    def prepare_metro_volume(data_dir: Path) -> Optional[Path]:
        """Download and prepare Metro Volume dataset."""
        output_path = data_dir / "metro+interstate+traffic+volume" / "Metro_Interstate_Traffic_Volume.csv"

        if output_path.exists():
            print(f"[INFO] Metro Volume already exists at {output_path}")
            return output_path

        urls = DataDownloader.DATASET_URLS.get("metro_volume", [])
        if not urls:
            print("[ERROR] Metro Volume URL not configured")
            return None

        if DataDownloader._download_from_candidates(urls, output_path) and DataDownloader._verify_csv(output_path):
            return output_path
        return None

    @staticmethod
    def prepare_beijing_pm25(data_dir: Path) -> Optional[Path]:
        """Prepare Beijing PM2.5 dataset via manual placement guidance."""
        output_path = data_dir / "beijing+pm2+5+data" / "PRSA_data_2010.1.1-2014.12.31.csv"

        if output_path.exists():
            print(f"[INFO] Beijing PM2.5 already exists at {output_path}")
            return output_path

        print("[INFO] Beijing PM2.5 dataset not found locally.")
        print("[INFO] Please place the CSV file at:")
        print(f"       {output_path}")
        print("[INFO] The fine-tuning script can fall back to synthetic data if missing.")
        return None

    @staticmethod
    def prepare_environmental_sensor_telemetry(data_dir: Path) -> Optional[Path]:
        """Prepare Environmental Sensor Telemetry dataset."""
        output_path = data_dir / "env-sens-telem-data" / "iot_telemetry_data.csv"

        if output_path.exists():
            print(f"[INFO] Environmental Sensor Telemetry already exists at {output_path}")
            return output_path

        print("[INFO] Environmental Sensor Telemetry dataset not found locally.")
        print("[INFO] Please place the Kaggle CSV at:")
        print(f"       {output_path}")
        print("[INFO] The fine-tuning script can fall back to synthetic data if missing.")
        return None

    @staticmethod
    def prepare_timesfm_benchmark_dataset(data_dir: Path, dataset_name: str) -> Optional[Path]:
        """Prepare one benchmark dataset used by evaluate_checkpoints.py."""
        if dataset_name not in DataDownloader.BENCHMARK_OUTPUTS:
            print(f"[ERROR] Unknown benchmark dataset: {dataset_name}")
            return None

        output_path = data_dir / DataDownloader.BENCHMARK_OUTPUTS[dataset_name]
        if output_path.exists():
            print(f"[INFO] {dataset_name} already exists at {output_path}")
            return output_path

        if dataset_name == "Traffic":
            prepared = DataDownloader._prepare_traffic_from_metro(data_dir, output_path)
            if prepared is not None:
                return prepared

        urls = DataDownloader.DATASET_URLS.get(dataset_name, [])
        if urls and DataDownloader._download_from_candidates(urls, output_path):
            if DataDownloader._verify_csv(output_path):
                return output_path
            return None

        print(f"[WARN] Could not auto-download {dataset_name}.")
        print("[INFO] Please place the CSV file manually at:")
        print(f"       {output_path}")
        if dataset_name == "Traffic":
            print("[INFO] If you already have Metro Interstate Traffic Volume data, place it at:")
            print("       data/metro+interstate+traffic+volume/Metro_Interstate_Traffic_Volume.csv")
            print("[INFO] Re-run with --traffic and the script will build data/traffic.csv automatically.")
        print("[INFO] Expected filenames for benchmark evaluator:")
        print("       ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv, electricity.csv, traffic.csv, weather.csv")
        return None

    @staticmethod
    def prepare_timesfm_benchmarks(data_dir: Path) -> Dict[str, Optional[Path]]:
        """Prepare all benchmark datasets for evaluate_checkpoints.py."""
        out: Dict[str, Optional[Path]] = {}
        for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Weather"]:
            print(f"\n[INFO] Preparing benchmark dataset: {name}")
            out[name] = DataDownloader.prepare_timesfm_benchmark_dataset(data_dir, name)
        return out


def main() -> None:
    """Dataset preparation CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare datasets for TSFM workflows")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--all", action="store_true", help="Prepare all known datasets (including benchmarks)")

    # Existing workflow datasets.
    parser.add_argument("--metro-volume", action="store_true", help="Download Metro Volume")
    parser.add_argument("--beijing-pm25", action="store_true", help="Prepare Beijing PM2.5")
    parser.add_argument("--environmental-sensor", action="store_true", help="Prepare Environmental Sensor Telemetry")

    # Benchmark bundle.
    parser.add_argument(
        "--timesfm-benchmarks",
        action="store_true",
        help="Prepare ETTh1/ETTh2/ETTm1/ETTm2/Electricity/Traffic/Weather",
    )

    # Individual benchmark flags.
    parser.add_argument("--ett-h1", action="store_true", help="Prepare ETTh1.csv")
    parser.add_argument("--ett-h2", action="store_true", help="Prepare ETTh2.csv")
    parser.add_argument("--ett-m1", action="store_true", help="Prepare ETTm1.csv")
    parser.add_argument("--ett-m2", action="store_true", help="Prepare ETTm2.csv")
    parser.add_argument("--electricity", action="store_true", help="Prepare electricity.csv")
    parser.add_argument("--traffic", action="store_true", help="Prepare traffic.csv")
    parser.add_argument("--weather", action="store_true", help="Prepare weather.csv")

    args = parser.parse_args()

    selected = any(
        [
            args.all,
            args.metro_volume,
            args.beijing_pm25,
            args.environmental_sensor,
            args.timesfm_benchmarks,
            args.ett_h1,
            args.ett_h2,
            args.ett_m1,
            args.ett_m2,
            args.electricity,
            args.traffic,
            args.weather,
        ]
    )

    if not selected:
        print("[INFO] No dataset selected.")
        print("[INFO] Use --all or one of:")
        print("       --metro-volume --beijing-pm25 --environmental-sensor")
        print("       --timesfm-benchmarks")
        print("       --ett-h1 --ett-h2 --ett-m1 --ett-m2 --electricity --traffic --weather")
        return

    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.metro_volume:
        print("\n[INFO] Preparing Metro Volume...")
        DataDownloader.prepare_metro_volume(args.data_dir)

    if args.all or args.beijing_pm25:
        print("\n[INFO] Preparing Beijing PM2.5...")
        DataDownloader.prepare_beijing_pm25(args.data_dir)

    if args.all or args.environmental_sensor:
        print("\n[INFO] Preparing Environmental Sensor Telemetry...")
        DataDownloader.prepare_environmental_sensor_telemetry(args.data_dir)

    if args.all or args.timesfm_benchmarks:
        DataDownloader.prepare_timesfm_benchmarks(args.data_dir)

    individual_map = {
        "ETTh1": args.ett_h1,
        "ETTh2": args.ett_h2,
        "ETTm1": args.ett_m1,
        "ETTm2": args.ett_m2,
        "Electricity": args.electricity,
        "Traffic": args.traffic,
        "Weather": args.weather,
    }
    for name, enabled in individual_map.items():
        if enabled:
            print(f"\n[INFO] Preparing benchmark dataset: {name}")
            DataDownloader.prepare_timesfm_benchmark_dataset(args.data_dir, name)

    print("\n[INFO] Dataset preparation complete!")


if __name__ == "__main__":
    main()
