#!/bin/bash
# Quick-start script for TSFM fine-tuning pipeline
# Usage: bash quickstart.sh [--all|--et-h1|--metro|--pm25|--eval]
# bash quickstart.sh --all                # Full pipeline
# bash quickstart.sh --quick              # Fast test
# bash quickstart.sh --linear-probe-only  # Statistical validation
# bash quickstart.sh --eval               # Just analyze results

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "TSFM Fine-Tuning Pipeline - Quick Start"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${YELLOW}[ERROR] Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
echo -e "${BLUE}[INFO] Python version: ${PYTHON_VERSION}${NC}"

# Check pre-trained model
if [ ! -f "tsfm_pretrain.pt" ]; then
    echo -e "${YELLOW}[WARN] Pre-trained model not found: tsfm_pretrain.pt${NC}"
    echo -e "${YELLOW}       You need to run pre-training first or provide the model.${NC}"
fi

# Create directories
mkdir -p data
mkdir -p finetuning_results
mkdir -p checkpoints

# Parse arguments
if [ "$1" == "--all" ] || [ "$1" == "" ]; then
    echo ""
    echo -e "${GREEN}Step 1: Download Datasets${NC}"
    echo "=================================================="
    python prepare_datasets.py --all || echo -e "${YELLOW}[WARN] Some datasets could not be downloaded (will use synthetic data)${NC}"
    
    echo ""
    echo -e "${GREEN}Step 2: Run Fine-Tuning Pipeline${NC}"
    echo "=================================================="
    python finetune_forecasting.py --train-from-scratch
    
    echo ""
    echo -e "${GREEN}Step 3: Analyze Results${NC}"
    echo "=================================================="
    python evaluate_results.py
    
    echo ""
    echo -e "${GREEN}✓ Pipeline Complete!${NC}"
    echo "Results saved to: finetuning_results/"
    
elif [ "$1" == "--ett-h1" ]; then
    echo -e "${GREEN}Downloading ETTh1 Dataset${NC}"
    python prepare_datasets.py --ett-h1
    
elif [ "$1" == "--metro" ]; then
    echo -e "${GREEN}Downloading Metro Volume Dataset${NC}"
    python prepare_datasets.py --metro-volume
    
elif [ "$1" == "--pm25" ]; then
    echo -e "${GREEN}Preparing Beijing PM2.5 Dataset${NC}"
    python prepare_datasets.py --beijing-pm25
    
elif [ "$1" == "--finetune" ]; then
    echo -e "${GREEN}Running Fine-Tuning Pipeline${NC}"
    python finetune_forecasting.py --train-from-scratch
    
elif [ "$1" == "--eval" ]; then
    echo -e "${GREEN}Analyzing Results${NC}"
    python evaluate_results.py
    
elif [ "$1" == "--linear-probe-only" ]; then
    echo -e "${GREEN}Running Linear Probe Only (Fast)${NC}"
    python finetune_forecasting.py --linear-probe-epochs 5 --finetune-epochs 0
    
elif [ "$1" == "--quick" ]; then
    echo -e "${GREEN}Quick Test (Synthetic Data, 2 Epochs)${NC}"
    python finetune_forecasting.py \
        --linear-probe-epochs 2 \
        --finetune-epochs 2 \
        --batch-size 16
    
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: bash quickstart.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  --all                Run complete pipeline (download + fine-tune + eval)"
    echo "  --ett-h1            Download ETTh1 dataset"
    echo "  --metro             Download Metro Volume dataset"
    echo "  --pm25              Prepare Beijing PM2.5 dataset"
    echo "  --finetune          Run fine-tuning with from-scratch baseline"
    echo "  --eval              Analyze results and generate plots"
    echo "  --linear-probe-only Run linear probe validation (fast)"
    echo "  --quick             Quick test with synthetic data (2 epochs)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash quickstart.sh                # Full pipeline"
    echo "  bash quickstart.sh --quick        # Fast test"
    echo "  bash quickstart.sh --linear-probe-only  # Validate pre-training"
    
else
    echo -e "${YELLOW}[ERROR] Unknown command: $1${NC}"
    echo "Run 'bash quickstart.sh --help' for usage information"
    exit 1
fi

echo ""
echo "=================================================="
echo "For detailed documentation, see: FINETUNE_README.md"
echo "=================================================="
