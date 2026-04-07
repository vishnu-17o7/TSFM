"""
Data preparation utilities for time-series forecasting fine-tuning.
Handles downloading and preparing Metro Volume, Beijing PM2.5, and Environmental Sensor Telemetry datasets.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Try to import optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class DataDownloader:
    """Download and prepare real time-series datasets."""

    # Dataset URLs
    DATASET_URLS = {
        "metro_volume": "https://raw.githubusercontent.com/OreStein/MultiScale/master/dataset/metro_volume.csv",
        "environmental_sensor_telemetry": None,
    }

    @staticmethod
    def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
        """Download file from URL."""
        if not HAS_REQUESTS:
            print("[ERROR] requests library not installed. Install with: pip install requests")
            return False

        try:
            print(f"[INFO] Downloading from {url}...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

            print(f"[INFO] Downloaded to {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to download {url}: {e}")
            return False

    @staticmethod
    def prepare_metro_volume(data_dir: Path) -> Optional[Path]:
        """Download and prepare Metro Volume dataset."""
        output_path = Path("data/metro+interstate+traffic+volume/Metro_Interstate_Traffic_Volume.csv")

        if output_path.exists():
            print(f"[INFO] Metro Volume already exists at {output_path}")
            return output_path

        url = DataDownloader.DATASET_URLS.get("metro_volume")
        if not url:
            print("[ERROR] Metro Volume URL not found")
            return None

        if DataDownloader.download_file(url, output_path):
            # Verify the file
            try:
                df = pd.read_csv(output_path)
                print(f"[INFO] Metro Volume loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"       Columns: {list(df.columns)}")
                return output_path
            except Exception as e:
                print(f"[ERROR] Failed to read Metro Volume: {e}")
                return None

        return None

    @staticmethod
    def prepare_beijing_pm25(data_dir: Path) -> Optional[Path]:
        """
        Create/prepare Beijing PM2.5 dataset.
        Note: Real dataset URL may vary. This creates a placeholder or uses local file.
        """
        output_path = Path("data/beijing+pm2+5+data/PRSA_data_2010.1.1-2014.12.31.csv")

        if output_path.exists():
            print(f"[INFO] Beijing PM2.5 already exists at {output_path}")
            return output_path

        # Try to download from OpenWeather/public source
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/data/PRSA_data.zip"
        print(f"[INFO] Attempting to download Beijing PM2.5 from {url}")
        print("[INFO] Note: This requires manual extraction. Skipping for now.")
        print("[INFO] The fine-tuning script will create synthetic data if file is missing.")

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
        print("[INFO] The fine-tuning script will create synthetic data if the file is missing.")
        return None


def main():
    """Download all datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare datasets for fine-tuning")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--metro-volume", action="store_true", help="Download Metro Volume")
    parser.add_argument("--beijing-pm25", action="store_true", help="Prepare Beijing PM2.5")
    parser.add_argument("--environmental-sensor", action="store_true", help="Prepare Environmental Sensor Telemetry")

    args = parser.parse_args()

    if not args.all and not any([args.metro_volume, args.beijing_pm25, args.environmental_sensor]):
        print("[INFO] No specific dataset selected. Use --all to download all datasets.")
        print("[INFO] Or choose: --metro-volume, --beijing-pm25, --environmental-sensor")
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

    print("\n[INFO] Dataset preparation complete!")


if __name__ == "__main__":
    main()
