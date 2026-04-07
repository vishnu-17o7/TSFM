"""
Fine-tuning TSFM for Time-Series Forecasting
Implements: Data acquisition, architecture adaptation, linear probe, full fine-tuning, and evaluation
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# Handle PyTorch 2.6+ checkpoint loading
try:
    import torch.serialization
    import pathlib
    import types
    import sys

    def _ensure_pathlib_local():
        if hasattr(pathlib, '_local'):
            return pathlib._local
        if 'pathlib._local' in sys.modules:
            return sys.modules['pathlib._local']
        local_mod = types.ModuleType('pathlib._local')
        local_mod.Path = pathlib.Path
        local_mod.PosixPath = pathlib.PosixPath
        local_mod.WindowsPath = pathlib.WindowsPath
        sys.modules['pathlib._local'] = local_mod
        setattr(pathlib, '_local', local_mod)
        return local_mod

    safe_classes = []
    for cls_name in [
        'Path', 'PosixPath', 'WindowsPath',
        'PurePath', 'PurePosixPath', 'PureWindowsPath',
    ]:
        if hasattr(pathlib, cls_name):
            safe_classes.append(getattr(pathlib, cls_name))
    local_module = _ensure_pathlib_local()
    for cls_name in ['Path', 'PosixPath', 'WindowsPath']:
        if hasattr(local_module, cls_name):
            safe_classes.append(getattr(local_module, cls_name))
    if safe_classes:
        torch.serialization.add_safe_globals(safe_classes)
except Exception:
    pass

# Import components from train_tsfm.py
from train_tsfm import RevIN, PatchEmbedding, TSTransformerBackbone


# ============================================================================
# Dataset Classes
# ============================================================================

class TimeSeriesDataset(Dataset):
    """
    Supervised time-series dataset with context windows and forecast targets.
    Input: window of length context_length (L=512)
    Output: forecast of length forecast_horizon (H=24, 96, 192, etc.)
    """

    def __init__(
        self,
        data: np.ndarray,
        context_length: int = 512,
        forecast_horizon: int = 24,
        stride: int = 1,
    ):
        self.data = data.astype(np.float32)
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride

        # Calculate valid starting positions
        max_start = len(data) - context_length - forecast_horizon
        if max_start < 0:
            raise ValueError(
                f"Data length {len(data)} is too short for context_length={context_length} "
                f"and forecast_horizon={forecast_horizon}"
            )

        self.valid_starts = list(range(0, max_start + 1, stride))

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.valid_starts[idx]
        context = self.data[start : start + self.context_length]
        target = self.data[start + self.context_length : start + self.context_length + self.forecast_horizon]
        return torch.from_numpy(context).unsqueeze(-1), torch.from_numpy(target).unsqueeze(-1)


class TimeSeriesDataModule:
    """Handles loading and preprocessing real time-series datasets."""

    @staticmethod
    def load_metro_volume(data_dir: Optional[Path] = None) -> np.ndarray:
        """Load Metro Volume dataset."""
        if data_dir is None:
            data_dir = Path("data")

        csv_path = data_dir / "metro+interstate+traffic+volume" / "Metro_Interstate_Traffic_Volume.csv"
        if not csv_path.exists():
            print(f"[INFO] Creating synthetic Metro Volume data for demo...")
            return TimeSeriesDataModule._create_synthetic_metro(length=10000)
        else:
            df = pd.read_csv(csv_path)
            # Use traffic_volume column (last column)
            data = df['traffic_volume'].values.astype(np.float32)
            return data

    @staticmethod
    def load_beijing_pm25(data_dir: Optional[Path] = None) -> np.ndarray:
        """Load Beijing PM2.5 dataset."""
        if data_dir is None:
            data_dir = Path("data")

        csv_path = data_dir / "beijing+pm2+5+data" / "PRSA_data_2010.1.1-2014.12.31.csv"
        if not csv_path.exists():
            print(f"[INFO] Creating synthetic Beijing PM2.5 data for demo...")
            return TimeSeriesDataModule._create_synthetic_pm25(length=10000)
        else:
            df = pd.read_csv(csv_path, index_col=0)
            # Use pm2.5 column, handle NA values
            pm25_series = df['pm2.5'].fillna(method='ffill').fillna(method='bfill')
            data = pm25_series.values.astype(np.float32)
            return data

    @staticmethod
    def load_environmental_sensor_telemetry(data_dir: Optional[Path] = None) -> np.ndarray:
        """Load Environmental Sensor Telemetry dataset."""
        if data_dir is None:
            data_dir = Path("data")

        candidate_paths = [
            data_dir / "environmental_sensor_telemetry" / "environmental_sensor_telemetry.csv",
            data_dir / "env-sens-telem-data" / "iot_telemetry_data.csv",
            data_dir / "env-sens-telem-data" / "environmental_sensor_telemetry.csv",
        ]

        csv_path = None
        for path in candidate_paths:
            if path.exists():
                csv_path = path
                break

        if csv_path is None:
            print(f"[INFO] Creating synthetic Environmental Sensor Telemetry data for demo...")
            return TimeSeriesDataModule._create_synthetic_environmental_sensor(length=10000)

        print(f"[INFO] Loading Environmental Sensor Telemetry from {csv_path}")
        df = pd.read_csv(csv_path)
        if "temp" in df.columns:
            data = df["temp"].astype(np.float32).values
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise ValueError("No numeric columns found in Environmental Sensor Telemetry CSV")
            data = numeric_df.mean(axis=1).astype(np.float32).values
        return data

    @staticmethod
    def _create_synthetic_ett(length: int = 8760, seed: int = 42) -> np.ndarray:
        """Create synthetic ETT-like data."""
        rng = np.random.default_rng(seed)
        t = np.arange(length)
        # Seasonal + trend + noise
        seasonal = 15 * np.sin(2 * np.pi * t / 365.25)
        trend = 0.01 * t
        noise = rng.normal(0, 2, length)
        data = 20 + seasonal + trend + noise
        return data.astype(np.float32)

    @staticmethod
    def _create_synthetic_metro(length: int = 5000, seed: int = 42) -> np.ndarray:
        """Create synthetic metro volume data."""
        rng = np.random.default_rng(seed)
        t = np.arange(length)
        # Periodic with trend
        periodic = 1000 * np.sin(2 * np.pi * t / 48)  # Daily cycle
        trend = 0.1 * t
        noise = rng.normal(0, 50, length)
        data = 5000 + periodic + trend + noise
        return np.maximum(data, 100).astype(np.float32)

    @staticmethod
    def _create_synthetic_pm25(length: int = 5000, seed: int = 42) -> np.ndarray:
        """Create synthetic PM2.5 data."""
        rng = np.random.default_rng(seed)
        t = np.arange(length)
        # Seasonal variation
        seasonal = 30 * np.sin(2 * np.pi * t / 365)
        trend = 0.02 * t
        noise = rng.normal(0, 5, length)
        data = 100 + seasonal + trend + noise
        return np.maximum(data, 10).astype(np.float32)

    @staticmethod
    def _create_synthetic_environmental_sensor(length: int = 10000, seed: int = 42) -> np.ndarray:
        """Create synthetic environmental telemetry sensor data."""
        rng = np.random.default_rng(seed)
        t = np.arange(length)
        daily = 20 * np.sin(2 * np.pi * t / 24)
        weekly = 10 * np.sin(2 * np.pi * t / (24 * 7))
        trend = 0.005 * t
        noise = rng.normal(0, 3, length)
        data = 50 + daily + weekly + trend + noise
        return np.maximum(data, 0).astype(np.float32)

    @staticmethod
    def normalize_series(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Normalize series to mean 0, std 1."""
        mean = data.mean()
        std = data.std()
        if std < 1e-6:
            std = 1.0
        normalized = (data - mean) / std
        return normalized.astype(np.float32), mean, std

    @staticmethod
    def split_series(
        data: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Chronological split of time series.
        train_ratio: 0.7, val_ratio: 0.1, test_ratio: 0.2
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = data[:train_end]
        val = data[train_end:val_end]
        test = data[val_end:]

        return train, val, test


# ============================================================================
# Model Classes
# ============================================================================

class TSFMForecastingHead(nn.Module):
    """
    Forecasting head for TSFM.
    Takes encoder outputs and projects to forecast horizon.
    """

    def __init__(
        self,
        embed_dim: int,
        num_patches: int,
        forecast_horizon: int,
        pooling: str = "mean",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.forecast_horizon = forecast_horizon
        self.pooling = pooling

        if pooling == "mean":
            # Average pooling across patch dimension
            self.projection = nn.Linear(embed_dim, forecast_horizon)
        elif pooling == "flatten":
            # Flatten all patch embeddings
            self.projection = nn.Sequential(
                nn.Linear(embed_dim * num_patches, 512),
                nn.GELU(),
                nn.Linear(512, forecast_horizon),
            )
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_patches, embed_dim) - unmasked encoder outputs
        Returns:
            (batch, forecast_horizon, 1)
        """
        if self.pooling == "mean":
            # Average pool across patches: (batch, num_patches, embed_dim) -> (batch, embed_dim)
            x = x.mean(dim=1)
            # Project to forecast horizon: (batch, embed_dim) -> (batch, forecast_horizon)
            x = self.projection(x)
        elif self.pooling == "flatten":
            # Flatten: (batch, num_patches, embed_dim) -> (batch, num_patches * embed_dim)
            x = x.reshape(x.size(0), -1)
            # Project to forecast horizon
            x = self.projection(x)

        # Add feature dimension: (batch, forecast_horizon) -> (batch, forecast_horizon, 1)
        x = x.unsqueeze(-1)
        return x


class TSFMForForecasting(nn.Module):
    """
    TSFM adapted for forecasting.
    Uses pre-trained encoder + RevIN normalization + new forecasting head.
    """

    def __init__(
        self,
        context_length: int,
        patch_length: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        forecast_horizon: int,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.context_length = context_length
        self.patch_length = patch_length
        self.embed_dim = embed_dim
        self.forecast_horizon = forecast_horizon

        # Pre-trained components
        self.revin = RevIN(num_features=1)
        self.patch_embedding = PatchEmbedding(context_length, patch_length, embed_dim)
        self.transformer = TSTransformerBackbone(embed_dim, num_heads, num_layers, dropout=dropout)

        # Forecasting head
        num_patches = context_length // patch_length
        self.forecasting_head = TSFMForecastingHead(
            embed_dim=embed_dim,
            num_patches=num_patches,
            forecast_horizon=forecast_horizon,
            pooling=pooling,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, context_length, 1)
        Returns:
            (batch, forecast_horizon, 1)
        """
        # RevIN normalization
        x_norm = self.revin(x, mode="norm")
        x_series = x_norm.squeeze(-1)  # (batch, context_length)

        # Patch embedding
        x_embedded = self.patch_embedding(x_series)  # (batch, num_patches, embed_dim)

        # Transformer encoding
        x_encoded = self.transformer(x_embedded)  # (batch, num_patches, embed_dim)

        # Forecasting head
        forecast = self.forecasting_head(x_encoded)  # (batch, forecast_horizon, 1)

        # RevIN denormalization
        forecast = self.revin(forecast, mode="denorm")  # (batch, forecast_horizon, 1)

        return forecast

    def freeze_encoder(self):
        """Freeze all encoder components for linear probe."""
        for module in [self.revin, self.patch_embedding, self.transformer]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning."""
        for module in [self.revin, self.patch_embedding, self.transformer]:
            for param in module.parameters():
                param.requires_grad = True

    def load_pretrained(self, checkpoint_path: str, device: torch.device):
        """Load pre-trained weights from checkpoint."""
        print(f"[INFO] Loading pre-trained weights from {checkpoint_path}...")
        checkpoint = None
        
        try:
            # Try safe loading first (PyTorch 2.6+)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception as e1:
            try:
                # Fallback: use safe_globals context manager for pathlib objects
                import pathlib
                import torch.serialization
                
                # Allow pathlib classes that might be in the checkpoint
                import types
                import sys

                def _ensure_local():
                    if hasattr(pathlib, '_local'):
                        return pathlib._local
                    if 'pathlib._local' in sys.modules:
                        return sys.modules['pathlib._local']
                    local_mod = types.ModuleType('pathlib._local')
                    local_mod.Path = pathlib.Path
                    local_mod.PosixPath = pathlib.PosixPath
                    local_mod.WindowsPath = pathlib.WindowsPath
                    sys.modules['pathlib._local'] = local_mod
                    setattr(pathlib, '_local', local_mod)
                    return local_mod

                safe_classes = []
                for cls_name in [
                    'Path', 'PosixPath', 'WindowsPath',
                    'PurePath', 'PurePosixPath', 'PureWindowsPath',
                ]:
                    if hasattr(pathlib, cls_name):
                        safe_classes.append(getattr(pathlib, cls_name))
                local_module = _ensure_local()
                for cls_name in ['Path', 'PosixPath', 'WindowsPath']:
                    if hasattr(local_module, cls_name):
                        safe_classes.append(getattr(local_module, cls_name))

                with torch.serialization.safe_globals(safe_classes):
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    
            except Exception as e2:
                print(f"[WARN] Could not load pre-trained weights: {e2}")
                print(f"[WARN] Will train model from scratch")
                return

        if checkpoint is None:
            print(f"[WARN] Checkpoint is None, training from scratch")
            return

        # Extract state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Load only compatible weights
        model_state = self.state_dict()
        incompatible_keys = []
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
            else:
                incompatible_keys.append(key)

        if incompatible_keys:
            print(f"[WARN] Incompatible keys: {incompatible_keys[:5]}...")

        self.load_state_dict(model_state, strict=False)
        print(f"[INFO] Pre-trained weights loaded successfully")


# ============================================================================
# Training & Evaluation Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    max_steps: Optional[int] = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    count = 0

    pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True)
    for step, (batch_x, batch_y) in enumerate(pbar):
        if max_steps and step >= max_steps:
            break

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate model and return MSE and MAE.
    Returns: (mse, mae, loss)
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    count = 0

    for batch_x, batch_y in tqdm(val_loader, desc="Evaluating", dynamic_ncols=True):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        mse = ((pred - batch_y) ** 2).mean().item()
        mae = (torch.abs(pred - batch_y)).mean().item()

        total_loss += loss.item()
        total_mse += mse
        total_mae += mae
        count += 1

    return total_mse / count, total_mae / count, total_loss / count


def linear_probe_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
) -> Dict[str, List[float]]:
    """
    Train only the forecasting head with frozen encoder (linear probe).
    """
    print("\n" + "=" * 80)
    print("LINEAR PROBE TRAINING (Frozen Encoder)")
    print("=" * 80)

    model.freeze_encoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    history = {"train_loss": [], "val_mse": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_mae, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}"
        )

    return history


def full_finetuning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 5e-5,
) -> Dict[str, List[float]]:
    """
    Full fine-tuning with unfrozen encoder (lower learning rate).
    """
    print("\n" + "=" * 80)
    print("FULL FINE-TUNING (Unfrozen Encoder)")
    print("=" * 80)

    model.unfreeze_encoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_mse": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_mae, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}"
        )

    return history


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune TSFM for time-series forecasting")
    parser.add_argument("--pretrained-model", type=str, default="tsfm_pretrain.pt", help="Path to pretrained model")
    parser.add_argument("--context-length", type=int, default=512, help="Context window length")
    parser.add_argument("--patch-length", type=int, default=32, help="Patch length")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--forecast-horizon", type=int, default=24, help="Forecast horizon (24, 96, 192)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--linear-probe-epochs", type=int, default=10, help="Linear probe epochs")
    parser.add_argument("--finetune-epochs", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--linear-probe-lr", type=float, default=1e-3, help="Linear probe learning rate")
    parser.add_argument("--finetune-lr", type=float, default=5e-5, help="Fine-tuning learning rate")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "flatten"], help="Pooling method")
    parser.add_argument("--train-from-scratch", action="store_true", help="Train identical model from scratch for baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("finetuning_results"), help="Output directory")

    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    datasets_info = {}
    all_results = {}

    for dataset_name, loader_fn in [
        ("Metro", TimeSeriesDataModule.load_metro_volume),
        ("Beijing_PM25", TimeSeriesDataModule.load_beijing_pm25),
        ("Environmental_Sensor_Telemetry", TimeSeriesDataModule.load_environmental_sensor_telemetry),
    ]:
        print(f"\n[INFO] Loading {dataset_name}...")
        data = loader_fn(args.data_dir)
        data_norm, mean, std = TimeSeriesDataModule.normalize_series(data)
        train_data, val_data, test_data = TimeSeriesDataModule.split_series(
            data_norm, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2
        )

        print(f"  - Original length: {len(data)}")
        print(f"  - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        print(f"  - Mean: {mean:.4f}, Std: {std:.4f}")

        datasets_info[dataset_name] = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "mean": mean,
            "std": std,
            "original": data,
        }

    # Train and evaluate on each dataset
    print("\n" + "=" * 80)
    print("FINE-TUNING ON DATASETS")
    print("=" * 80)

    for dataset_name, data_dict in datasets_info.items():
        print(f"\n{'=' * 80}")
        print(f"DATASET: {dataset_name}")
        print(f"{'=' * 80}")

        dataset_results = {}

        # Create datasets
        train_dataset = TimeSeriesDataset(
            data_dict["train"],
            context_length=args.context_length,
            forecast_horizon=args.forecast_horizon,
            stride=args.context_length // 4,  # 25% overlap
        )
        val_dataset = TimeSeriesDataset(
            data_dict["val"],
            context_length=args.context_length,
            forecast_horizon=args.forecast_horizon,
            stride=args.context_length,  # No overlap for validation
        )
        test_dataset = TimeSeriesDataset(
            data_dict["test"],
            context_length=args.context_length,
            forecast_horizon=args.forecast_horizon,
            stride=args.context_length,  # No overlap for test
        )

        print(f"  Train windows: {len(train_dataset)}")
        print(f"  Val windows: {len(val_dataset)}")
        print(f"  Test windows: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # ---- Pretrained Fine-tuning ----
        print(f"\n[INFO] Training with pre-trained model...")
        model = TSFMForForecasting(
            context_length=args.context_length,
            patch_length=args.patch_length,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            forecast_horizon=args.forecast_horizon,
            pooling=args.pooling,
        ).to(device)

        # Load pre-trained weights
        if Path(args.pretrained_model).exists():
            model.load_pretrained(args.pretrained_model, device)
        else:
            print(f"[WARN] Pre-trained model not found: {args.pretrained_model}")
            print(f"[INFO] Training from scratch instead")

        # Linear probe
        linear_probe_history = linear_probe_training(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.linear_probe_epochs,
            lr=args.linear_probe_lr,
        )

        dataset_results["linear_probe"] = linear_probe_history

        # Full fine-tuning
        finetune_history = full_finetuning(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
        )

        dataset_results["finetune"] = finetune_history

        # Evaluate on test set
        print(f"\n[INFO] Evaluating on test set...")
        criterion = nn.MSELoss()
        test_mse, test_mae, test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Test MAE: {test_mae:.6f}")

        dataset_results["test_mse"] = test_mse
        dataset_results["test_mae"] = test_mae

        # Save model
        model_path = args.output_dir / f"model_pretrained_{dataset_name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Model saved to {model_path}")

        # ---- From-Scratch Baseline ----
        if args.train_from_scratch:
            print(f"\n[INFO] Training identical model from scratch (baseline)...")
            baseline_model = TSFMForForecasting(
                context_length=args.context_length,
                patch_length=args.patch_length,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                forecast_horizon=args.forecast_horizon,
                pooling=args.pooling,
            ).to(device)

            # Train without linear probe (full model from scratch)
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-4, weight_decay=1e-4)

            scratch_history = {"train_loss": [], "val_mse": [], "val_mae": []}

            for epoch in range(1, args.finetune_epochs + 1):
                train_loss = train_epoch(baseline_model, train_loader, optimizer, criterion, device)
                val_mse, val_mae, _ = evaluate(baseline_model, val_loader, criterion, device)

                scratch_history["train_loss"].append(train_loss)
                scratch_history["val_mse"].append(val_mse)
                scratch_history["val_mae"].append(val_mae)

                print(
                    f"Epoch {epoch}/{args.finetune_epochs} | Train Loss: {train_loss:.6f} | "
                    f"Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}"
                )

            test_mse_scratch, test_mae_scratch, _ = evaluate(baseline_model, test_loader, criterion, device)
            print(f"Scratch Test MSE: {test_mse_scratch:.6f}")
            print(f"Scratch Test MAE: {test_mae_scratch:.6f}")

            dataset_results["scratch"] = scratch_history
            dataset_results["scratch_test_mse"] = test_mse_scratch
            dataset_results["scratch_test_mae"] = test_mae_scratch

            # Save baseline model
            baseline_model_path = args.output_dir / f"model_scratch_{dataset_name}.pt"
            torch.save(baseline_model.state_dict(), baseline_model_path)
            print(f"[INFO] Baseline model saved to {baseline_model_path}")

        all_results[dataset_name] = dataset_results

    # Save results
    print(f"\n[INFO] Saving results...")
    results_summary = {}
    for dataset_name, results in all_results.items():
        results_summary[dataset_name] = {
            "test_mse": results.get("test_mse"),
            "test_mae": results.get("test_mae"),
            "scratch_test_mse": results.get("scratch_test_mse"),
            "scratch_test_mae": results.get("scratch_test_mae"),
        }

    results_file = args.output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[INFO] Results saved to {results_file}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for dataset_name, results in results_summary.items():
        print(f"\n{dataset_name}:")
        print(f"  Pretrained - Test MSE: {results['test_mse']:.6f}, MAE: {results['test_mae']:.6f}")
        if results['scratch_test_mse'] is not None:
            print(f"  From-Scratch - Test MSE: {results['scratch_test_mse']:.6f}, MAE: {results['scratch_test_mae']:.6f}")
            improvement = (results['scratch_test_mse'] - results['test_mse']) / results['scratch_test_mse'] * 100
            print(f"  Improvement: {improvement:.2f}%")


if __name__ == "__main__":
    main()
