"""
Download real-world time-series pretraining corpora.

Supported sources:
  - Monash Forecasting Repository: ~30 datasets in .tsf format
  - M4 Competition: Annual/Quarterly/Monthly/Weekly/Daily/Hourly series

Outputs to --output-dir (default: data/real_corpora) with a manifest.json tracking provenance.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    from tqdm import tqdm as tqdm_bar
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm_bar(iterable, **kwargs):
        return iterable


def _download_file(url: str, dest: Path, chunk_size: int = 65536, timeout: int = 120) -> bool:
    """Download a file with progress, returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            if HAS_TQDM and total > 0:
                with tqdm_bar(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"[WARN] Failed to download {url}: {e}")
        return False


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


MONASH_DATASETS = [
    {"name": "australian_electricity_demand", "url": "https://zenodo.org/records/4656149/files/australian_electricity_demand_dataset.zip"},
    {"name": "car_parts", "url": "https://zenodo.org/records/4656120/files/car_parts_dataset.zip"},
    {"name": "cif_2016", "url": "https://zenodo.org/records/4656040/files/cif_2016_dataset.zip"},
    {"name": "covid_deaths", "url": "https://zenodo.org/records/4656009/files/covid_deaths_dataset.zip"},
    {"name": "dominick", "url": "https://zenodo.org/records/4654802/files/dominick_dataset.zip"},
    {"name": "electricity_hourly", "url": "https://zenodo.org/records/4656140/files/electricity_hourly_dataset.zip"},
    {"name": "electricity_weekly", "url": "https://zenodo.org/records/4656144/files/electricity_weekly_dataset.zip"},
    {"name": "fred_md", "url": "https://zenodo.org/records/4654833/files/fred_md_dataset.zip"},
    {"name": "hospital", "url": "https://zenodo.org/records/4656014/files/hospital_dataset.zip"},
    {"name": "kaggle_web_traffic", "url": "https://zenodo.org/records/4656080/files/kaggle_web_traffic_dataset.zip"},
    {"name": "kaggle_web_traffic_weekly", "url": "https://zenodo.org/records/4656664/files/kaggle_web_traffic_weekly_dataset.zip"},
    {"name": "kdd_cup_2018", "url": "https://zenodo.org/records/4656719/files/kdd_cup_2018_dataset.zip"},
    {"name": "london_smart_meters", "url": "https://zenodo.org/records/4656072/files/london_smart_meters_dataset.zip"},
    {"name": "m1_monthly", "url": "https://zenodo.org/records/4656156/files/m1_monthly_dataset.zip"},
    {"name": "m1_quarterly", "url": "https://zenodo.org/records/4656159/files/m1_quarterly_dataset.zip"},
    {"name": "m1_yearly", "url": "https://zenodo.org/records/4656193/files/m1_yearly_dataset.zip"},
    {"name": "m3_monthly", "url": "https://zenodo.org/records/4656298/files/m3_monthly_dataset.zip"},
    {"name": "m3_other", "url": "https://zenodo.org/records/4656337/files/m3_other_dataset.zip"},
    {"name": "m3_quarterly", "url": "https://zenodo.org/records/4656262/files/m3_quarterly_dataset.zip"},
    {"name": "m3_yearly", "url": "https://zenodo.org/records/4656222/files/m3_yearly_dataset.zip"},
    {"name": "m4_daily", "url": "https://zenodo.org/records/4656548/files/m4_daily_dataset.zip"},
    {"name": "m4_hourly", "url": "https://zenodo.org/records/4656589/files/m4_hourly_dataset.zip"},
    {"name": "m4_monthly", "url": "https://zenodo.org/records/4656379/files/m4_monthly_dataset.zip"},
    {"name": "m4_quarterly", "url": "https://zenodo.org/records/4656416/files/m4_quarterly_dataset.zip"},
    {"name": "m4_weekly", "url": "https://zenodo.org/records/4656522/files/m4_weekly_dataset.zip"},
    {"name": "m4_yearly", "url": "https://zenodo.org/records/4656380/files/m4_yearly_dataset.zip"},
    {"name": "nn5_daily", "url": "https://zenodo.org/records/4656110/files/nn5_daily_dataset.zip"},
    {"name": "nn5_weekly", "url": "https://zenodo.org/records/4656127/files/nn5_weekly_dataset.zip"},
    {"name": "pedestrian_counts", "url": "https://zenodo.org/records/4656626/files/pedestrian_counts_dataset.zip"},
    {"name": "solar_10_minutes", "url": "https://zenodo.org/records/4656147/files/solar_10_minutes_dataset.zip"},
    {"name": "solar_weekly", "url": "https://zenodo.org/records/4656151/files/solar_weekly_dataset.zip"},
    {"name": "tourism_monthly", "url": "https://zenodo.org/records/4656096/files/tourism_monthly_dataset.zip"},
    {"name": "tourism_quarterly", "url": "https://zenodo.org/records/4656093/files/tourism_quarterly_dataset.zip"},
    {"name": "tourism_yearly", "url": "https://zenodo.org/records/4656103/files/tourism_yearly_dataset.zip"},
    {"name": "traffic_hourly", "url": "https://zenodo.org/records/4656132/files/traffic_hourly_dataset.zip"},
    {"name": "traffic_weekly", "url": "https://zenodo.org/records/4656135/files/traffic_weekly_dataset.zip"},
    {"name": "weather", "url": "https://zenodo.org/records/4654822/files/weather_dataset.zip"},
    {"name": "wind_farms", "url": "https://zenodo.org/records/4654858/files/wind_farms_dataset.zip"},
]


def _unzip_tsf_files(zip_path: Path, out_dir: Path) -> List[Path]:
    tsf_files: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith(".tsf") or member.endswith(".ts"):
                target = out_dir / Path(member).name
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                tsf_files.append(target)
    return tsf_files


def download_monash(output_dir: Path, datasets: Optional[List[str]] = None,
                    max_datasets: Optional[int] = None) -> Dict[str, List[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, List[Path]] = {}

    selected = MONASH_DATASETS
    if datasets:
        name_set = set(datasets)
        selected = [d for d in MONASH_DATASETS if d["name"] in name_set]

    if max_datasets is not None:
        selected = selected[:max_datasets]

    print(f"[INFO] Downloading {len(selected)} Monash datasets...")
    for ds in selected:
        name = ds["name"]
        url = ds["url"]
        dest_dir = output_dir / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        existing_tsf = list(dest_dir.rglob("*.tsf")) + list(dest_dir.rglob("*.ts"))
        if existing_tsf:
            print(f"[SKIP] {name} already downloaded ({len(existing_tsf)} .tsf files)")
            result[name] = existing_tsf
            continue

        zip_path = dest_dir / f"{name}.zip"
        print(f"[DOWNLOAD] {name} <- {url}")
        if not _download_file(url, zip_path):
            print(f"[FAIL] {name}: download failed")
            continue

        tsf_files = _unzip_tsf_files(zip_path, dest_dir)
        if not tsf_files:
            print(f"[WARN] {name}: no .tsf/.ts files found in zip")
            continue

        try:
            zip_path.unlink()
        except Exception:
            pass

        result[name] = tsf_files
        print(f"[OK] {name}: {len(tsf_files)} .tsf files")

    return result


def download_m4(output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_tsf: List[Path] = []
    m4_datasets = [d for d in MONASH_DATASETS if d["name"].startswith("m4_")]
    for ds in m4_datasets:
        r = download_monash(output_dir, datasets=[ds["name"]])
        for files in r.values():
            all_tsf.extend(files)
    return all_tsf


def write_manifest(output_dir: Path, results: Dict[str, List[Path]]) -> Path:
    manifest = {
        "source": "monash_forecasting_repository",
        "url": "https://zenodo.org/communities/monash-ts-forecasting",
        "datasets": {},
    }
    total_tsf = 0
    for name, files in results.items():
        manifest["datasets"][name] = {
            "num_files": len(files),
            "files": [str(f.relative_to(output_dir)) if f.is_relative_to(output_dir) else str(f) for f in files],
        }
        total_tsf += len(files)
    manifest["total_tsf_files"] = total_tsf

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Manifest written to {manifest_path} ({total_tsf} total .tsf files)")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real-world pretraining corpora for TSFM.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/real_corpora"),
                        help="Directory to store downloaded corpora.")
    parser.add_argument("--source", choices=["monash", "m4", "all"], default="all",
                        help="Which corpora to download.")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific Monash dataset names to download (default: all).")
    parser.add_argument("--max-datasets", type=int, default=0,
                        help="Maximum number of Monash datasets to download (0 = all).")
    parser.add_argument("--no-manifest", action="store_true",
                        help="Skip writing manifest.json.")

    args = parser.parse_args()
    max_ds = args.max_datasets if args.max_datasets > 0 else None

    results: Dict[str, List[Path]] = {}

    if args.source in ("monash", "all"):
        print("[INFO] Downloading Monash Forecasting Repository datasets...")
        monash_results = download_monash(args.output_dir, datasets=args.datasets, max_datasets=max_ds)
        results.update(monash_results)

    if args.source in ("m4", "all") and not args.datasets:
        print("[INFO] Downloading M4 Competition datasets...")
        m4_results = download_m4(args.output_dir)
        seen = set()
        for name in results:
            for f in results[name]:
                seen.add(f.name)
        m4_merged = {}
        for name, files in {"m4_combined": [f for f in m4_results if f.name not in seen]}.items():
            if files:
                m4_merged[name] = files
        results.update(m4_merged)

    if not results:
        print("[WARN] No datasets were downloaded. Check internet connection and URLs.")
        return

    if not args.no_manifest:
        write_manifest(args.output_dir, results)

    total_files = sum(len(f) for f in results.values())
    print(f"\n[INFO] Done. Downloaded {len(results)} datasets ({total_files} .tsf files) to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
