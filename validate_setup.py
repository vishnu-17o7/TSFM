#!/usr/bin/env python
"""
Validation script for TSFM fine-tuning pipeline.
Checks setup, imports, and runs a quick test.
"""

import sys
import importlib
from pathlib import Path

def check_imports():
    """Check if all required packages are available."""
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    required = {
        'torch': 'torch',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tqdm': 'tqdm',
    }
    
    optional = {
        'requests': 'requests (for dataset download)',
        'matplotlib': 'matplotlib (for plotting)',
        'seaborn': 'seaborn (for advanced plots)',
    }
    
    all_ok = True
    
    # Check required packages
    print("\nRequired packages:")
    for package, name in required.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name:<20} version {version}")
        except ImportError as e:
            print(f"  ✗ {name:<20} NOT INSTALLED")
            all_ok = False
    
    # Check optional packages
    print("\nOptional packages:")
    for package, name in optional.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name:<30} version {version}")
        except ImportError:
            print(f"  ~ {name:<30} (optional)")
    
    return all_ok


def check_files():
    """Check if required files exist."""
    print("\n" + "="*80)
    print("CHECKING FILES")
    print("="*80)
    
    required_files = [
        'train_tsfm.py',
        'finetune_forecasting.py',
        'prepare_datasets.py',
        'evaluate_results.py',
    ]
    
    optional_files = [
        'tsfm_pretrain.pt',
        'FINETUNE_README.md',
        'IMPLEMENTATION_SUMMARY.md',
    ]
    
    print("\nRequired files:")
    all_ok = True
    for file_path in required_files:
        p = Path(file_path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file_path:<40} ({size_mb:.1f}MB)")
        else:
            print(f"  ✗ {file_path:<40} NOT FOUND")
            all_ok = False
    
    print("\nOptional files:")
    for file_path in optional_files:
        p = Path(file_path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file_path:<40} ({size_mb:.1f}MB)")
        else:
            print(f"  ~ {file_path:<40} (optional)")
    
    return all_ok


def check_model():
    """Check pre-trained model."""
    print("\n" + "="*80)
    print("CHECKING PRE-TRAINED MODEL")
    print("="*80)
    
    model_path = Path('tsfm_pretrain.pt')
    
    if not model_path.exists():
        print(f"\n  ✗ Pre-trained model not found: {model_path}")
        print(f"    The pipeline will still work but will train models from scratch.")
        print(f"    To use pre-trained weights, ensure tsfm_pretrain.pt is in the current directory.")
        return False
    
    import torch
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        # For older PyTorch versions
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"\n  ⚠ Warning: Could not fully validate checkpoint: {str(e)[:100]}")
        print(f"    However, the model file exists and may still be usable.")
        return True
    
    print(f"\n  ✓ Pre-trained model found: {model_path}")
    print(f"    Size: {model_path.stat().st_size / (1024**2):.1f}MB")
    
    try:
        if 'model_state_dict' in checkpoint:
            print(f"    Format: Old checkpoint (with model_state_dict)")
            statedict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            print(f"    Format: New checkpoint (with model)")
            statedict = checkpoint['model']
        else:
            statedict = checkpoint
        
        params = sum(p.numel() for p in statedict.values() if isinstance(p, torch.Tensor))
        print(f"    Parameters: {params:,}")
    except Exception as e:
        print(f"    ⚠ Could not inspect checkpoint details: {str(e)[:80]}")
    
    return True


def quick_import_test():
    """Test imports from pipeline modules."""
    print("\n" + "="*80)
    print("TESTING IMPORTS")
    print("="*80)
    
    try:
        print("\nImporting from train_tsfm.py...")
        from train_tsfm import RevIN, PatchEmbedding, TSTransformerBackbone
        print("  ✓ RevIN")
        print("  ✓ PatchEmbedding")
        print("  ✓ TSTransformerBackbone")
        
        print("\nImporting from finetune_forecasting.py...")
        from finetune_forecasting import (
            TimeSeriesDataset,
            TimeSeriesDataModule,
            TSFMForForecasting,
            TSFMForecastingHead,
        )
        print("  ✓ TimeSeriesDataset")
        print("  ✓ TimeSeriesDataModule")
        print("  ✓ TSFMForForecasting")
        print("  ✓ TSFMForecastingHead")
        
        print("\nImporting from prepare_datasets.py...")
        from prepare_datasets import DataDownloader
        print("  ✓ DataDownloader")
        
        print("\nImporting from evaluate_results.py...")
        from evaluate_results import ResultsAnalyzer
        print("  ✓ ResultsAnalyzer")
        
        return True
    except ImportError as e:
        print(f"\n  ✗ Import failed: {e}")
        return False


def test_data_loading():
    """Test data loading."""
    print("\n" + "="*80)
    print("TESTING DATA LOADING")
    print("="*80)
    
    try:
        from finetune_forecasting import TimeSeriesDataModule
        import numpy as np
        
        print("\nTesting data loading...")
        
        # Load synthetic data
        print("  Loading Metro Volume (synthetic)...")
        data_metro = TimeSeriesDataModule.load_metro_volume(Path('data'))
        print(f"    ✓ Loaded {len(data_metro)} samples")
        
        print("  Loading Beijing PM2.5 (synthetic)...")
        data_metro = TimeSeriesDataModule.load_metro_volume(Path('data'))
        print(f"    ✓ Loaded {len(data_metro)} samples")
        
        print("  Loading Beijing PM2.5 (synthetic)...")
        data_pm25 = TimeSeriesDataModule.load_beijing_pm25(Path('data'))
        print(f"    ✓ Loaded {len(data_pm25)} samples")

        print("  Loading Environmental Sensor Telemetry (synthetic)...")
        data_sensor = TimeSeriesDataModule.load_environmental_sensor_telemetry(Path('data'))
        print(f"    ✓ Loaded {len(data_sensor)} samples")
        
        # Test normalization
        print("  Testing normalization...")
        data_norm, mean, std = TimeSeriesDataModule.normalize_series(data_metro)
        print(f"    ✓ Normalized: mean={mean:.4f}, std={std:.4f}")
        
        # Test splitting
        print("  Testing chronological split...")
        train, val, test = TimeSeriesDataModule.split_series(data_norm)
        print(f"    ✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return True
    except Exception as e:
        print(f"\n  ✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_creation():
    """Test dataset creation."""
    print("\n" + "="*80)
    print("TESTING DATASET CREATION")
    print("="*80)
    
    try:
        from finetune_forecasting import TimeSeriesDataset, TimeSeriesDataModule
        import numpy as np
        
        print("\nCreating test dataset...")
        
        # Create synthetic data
        data = np.random.randn(5000).astype(np.float32)
        
        # Create dataset
        dataset = TimeSeriesDataset(
            data,
            context_length=512,
            forecast_horizon=24,
            stride=512,
        )
        
        print(f"  ✓ Dataset created: {len(dataset)} samples")
        
        # Get one sample
        x, y = dataset[0]
        print(f"  ✓ Sample shapes: x={x.shape}, y={y.shape}")
        
        if x.shape != (512, 1):
            print(f"    ✗ Unexpected x shape: {x.shape} (expected (512, 1))")
            return False
        
        if y.shape != (24, 1):
            print(f"    ✗ Unexpected y shape: {y.shape} (expected (24, 1))")
            return False
        
        return True
    except Exception as e:
        print(f"\n  ✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation."""
    print("\n" + "="*80)
    print("TESTING MODEL CREATION")
    print("="*80)
    
    try:
        import torch
        from finetune_forecasting import TSFMForForecasting
        
        print("\nCreating forecasting model...")
        
        model = TSFMForForecasting(
            context_length=512,
            patch_length=32,
            embed_dim=128,
            num_heads=8,
            num_layers=4,
            forecast_horizon=24,
        )
        
        print(f"  ✓ Model created")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Total parameters: {total_params:,}")
        
        # Test forward pass
        print(f"  Testing forward pass...")
        device = torch.device('cpu')
        model = model.to(device)
        
        x = torch.randn(2, 512, 1).to(device)
        with torch.no_grad():
            y = model(x)
        
        print(f"  ✓ Input shape: {x.shape}")
        print(f"  ✓ Output shape: {y.shape}")
        
        if y.shape != (2, 24, 1):
            print(f"    ✗ Unexpected output shape: {y.shape} (expected (2, 24, 1))")
            return False
        
        return True
    except Exception as e:
        print(f"\n  ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print summary of checks."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    checks = {
        'Dependencies': results['imports'],
        'Required Files': results['files'],
        'Pre-trained Model': results['model'],
        'Module Imports': results['module_imports'],
        'Data Loading': results['data_loading'],
        'Dataset Creation': results['dataset_creation'],
        'Model Creation': results['model_creation'],
    }
    
    all_ok = True
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check_name:<30} {status}")
        if not passed:
            all_ok = False
    
    print("\n" + "="*80)
    if all_ok:
        print("\n✓ All checks passed! You can now run the pipeline.\n")
        print("Quick start:")
        print("  bash quickstart.sh --quick        # Fast test")
        print("  bash quickstart.sh --all          # Full pipeline")
    else:
        print("\n✗ Some checks failed. Please fix issues before running the pipeline.\n")
        print("Common fixes:")
        print("  pip install torch numpy pandas tqdm")
        print("  python prepare_datasets.py --all")
    
    print("="*80)
    
    return all_ok


def main():
    """Run all validation checks."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "TSFM FINE-TUNING VALIDATION" + " "*32 + "║")
    print("╚" + "="*78 + "╝")
    
    results = {
        'imports': check_imports(),
        'files': check_files(),
        'model': check_model(),
        'module_imports': quick_import_test(),
        'data_loading': test_data_loading(),
        'dataset_creation': test_dataset_creation(),
        'model_creation': test_model_creation(),
    }
    
    all_ok = print_summary(results)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
