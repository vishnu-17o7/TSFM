#!/bin/bash
# TSFM Quick-Start Script
# ========================
# Full pipeline: pretrain → ablation → fine-tune → benchmark → evaluate
#
# Usage:
#   bash quickstart.sh                     # Full pipeline (default)
#   bash quickstart.sh --pretrain          # Pretrain only
#   bash quickstart.sh --ablation          # Multi-seed ablation only
#   bash quickstart.sh --finetune          # Fine-tune only
#   bash quickstart.sh --benchmark-eval    # Benchmark TSFM vs TimesFM
#   bash quickstart.sh --eval             # Analyze fine-tuning results
#   bash quickstart.sh --quick            # Fast smoke test (2 epochs)
#   bash quickstart.sh --help             # Show this help

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "TSFM Pipeline - Quick Start"
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

# Create directories
mkdir -p data
mkdir -p data/real_corpora
mkdir -p finetuning_results
mkdir -p checkpoints
mkdir -p experiments/multiseed

# ============================================================
# Pipeline Commands
# ============================================================

if [ "$1" == "--all" ] || [ "$1" == "" ]; then
    echo ""
    echo -e "${GREEN}Step 1: Download Real-World Pretraining Corpora${NC}"
    echo "=================================================="
    python download_pretraining_corpora.py \
        --output-dir data/real_corpora \
        --max-datasets 10 \
        || echo -e "${YELLOW}[WARN] Some corpora could not be downloaded${NC}"

    echo ""
    echo -e "${GREEN}Step 2: Prepare Benchmark Datasets (7/7)${NC}"
    echo "=================================================="
    python prepare_datasets.py --timesfm-benchmarks \
        || echo -e "${YELLOW}[WARN] Some datasets could not be downloaded (will use synthetic fallback)${NC}"

    echo ""
    echo -e "${GREEN}Step 3: Pretrain TSFM (runs until convergence)${NC}"
    echo "=================================================="
    echo -e "${BLUE}  Loss: Huber | Augmentation: ON | EMA: 0.999 | LR: Cosine w/ warmup${NC}"
    echo -e "${BLUE}  Epochs: 100 (early stopping patience=10 — will stop when converged)${NC}"
    python train_tsfm.py \
        --data-dir data \
        --real-data-dir data/real_corpora \
        --epochs 100 \
        --batch-size 8192 \
        --gradient-accumulation-steps 1 \
        --num-workers 32 \
        --loss-fn huber \
        --ema-decay 0.999 \
        --augment \
        --early-stopping-patience 10 \
        --best-model-path tsfm_best.pt \
        --metrics-out experiments/train_metrics.json

    echo ""
    echo -e "${GREEN}Step 4: Multi-Seed Ablation (7 seeds, convergence per run)${NC}"
    echo "=================================================="
    echo -e "${BLUE}  Seeds: 11,42,123,256,512,789,1024 | CI: t + bootstrap | Welch + Holm${NC}"
    echo -e "${BLUE}  Each run: up to 100 epochs, early stopping patience=10${NC}"
    python run_multiseed_ablation.py \
        --data-dir data \
        --real-data-dir data/real_corpora \
        --seeds 11,42,123,256,512,789,1024 \
        --epochs 100 \
        --batch-size 8192 \
        --gradient-accumulation-steps 1 \
        --num-workers 32 \
        --early-stopping-patience 10 \
        --loss-fn huber \
        --ema-decay 0.999 \
        --ci-method both

    echo ""
    echo -e "${GREEN}Step 5: Visualize Training Curves${NC}"
    echo "=================================================="
    python experiments/training_curves.py \
        --metrics-dir experiments/multiseed \
        --output experiments/training_curves.png

    echo ""
    echo -e "${GREEN}Step 6: Estimate Required Seeds${NC}"
    echo "=================================================="
    python experiments/seed_count_planner.py \
        --pilot-csv experiments/multiseed_summary.csv

    echo ""
    echo -e "${GREEN}Step 7: Fine-Tune for Forecasting${NC}"
    echo "=================================================="
    python finetune_forecasting.py --train-from-scratch

    echo ""
    echo -e "${GREEN}Step 8: Benchmark Evaluation (TSFM vs TimesFM)${NC}"
    echo "=================================================="
    python evaluate_checkpoints.py \
        --models both \
        --checkpoint-glob "tsfm_best.pt" \
        --forecast-horizon 96 \
        --datasets ETTh1 ETTh2 ETTm1 ETTm2 Electricity Traffic Weather

    echo ""
    echo -e "${GREEN}Step 9: Print Leaderboard${NC}"
    echo "=================================================="
    python benchmark_leaderboard.py \
        --input finetuning_results/benchmark_results.json \
        --sort-by mean_mse \
        --show-dataset-winners

    echo ""
    echo -e "${GREEN}Step 10: Analyze Fine-Tuning Results${NC}"
    echo "=================================================="
    python evaluate_results.py

    echo ""
    echo -e "${GREEN}✓ Full Pipeline Complete!${NC}"

# ── Individual Commands ──────────────────────────────────────

elif [ "$1" == "--pretrain" ]; then
    echo -e "${GREEN}Pretraining TSFM until convergence (Huber + EMA + Augmentation)${NC}"
    python train_tsfm.py \
        --data-dir data \
        --real-data-dir data/real_corpora \
        --epochs 100 \
        --batch-size 8192 \
        --gradient-accumulation-steps 1 \
        --num-workers 32 \
        --loss-fn huber \
        --ema-decay 0.999 \
        --augment \
        --early-stopping-patience 10 \
        --best-model-path tsfm_best.pt \
        --metrics-out experiments/train_metrics.json

elif [ "$1" == "--pretrain-memory-safe" ]; then
    echo -e "${GREEN}Pretraining TSFM until convergence (Gradient Checkpointing)${NC}"
    python train_tsfm.py \
        --data-dir data \
        --real-data-dir data/real_corpora \
        --epochs 100 \
        --loss-fn huber \
        --ema-decay 0.999 \
        --augment \
        --gradient-checkpointing \
        --batch-size 16 \
        --early-stopping-patience 10 \
        --best-model-path tsfm_best.pt \
        --metrics-out experiments/train_metrics.json

elif [ "$1" == "--pretrain-cpu" ]; then
    echo -e "${GREEN}Pretraining TSFM until convergence (CPU-safe mode)${NC}"
    python train_tsfm.py \
        --data-dir data \
        --epochs 100 \
        --loss-fn huber \
        --no-augment \
        --no-amp \
        --no-compile \
        --batch-size 16 \
        --num-workers 4 \
        --feature-workers 4 \
        --early-stopping-patience 10 \
        --best-model-path tsfm_best.pt \
        --metrics-out experiments/train_metrics.json

elif [ "$1" == "--ablation" ]; then
    echo -e "${GREEN}Running Multi-Seed Ablation (7 seeds, convergence per run)${NC}"
    python run_multiseed_ablation.py \
        --data-dir data \
        --real-data-dir data/real_corpora \
        --seeds 11,42,123,256,512,789,1024 \
        --epochs 100 \
        --batch-size 8192 \
        --gradient-accumulation-steps 1 \
        --num-workers 32 \
        --early-stopping-patience 10 \
        --loss-fn huber \
        --ema-decay 0.999 \
        --ci-method both

elif [ "$1" == "--ablation-pilot" ]; then
    echo -e "${GREEN}Running Pilot Ablation (3 seeds + seed estimation)${NC}"
    python run_multiseed_ablation.py \
        --data-dir data \
        --seeds 11,42,123 \
        --epochs 5 \
        --loss-fn huber \
        --estimate-seeds \
        --target-ci-half-width-ratio 0.10

elif [ "$1" == "--finetune" ]; then
    echo -e "${GREEN}Running Fine-Tuning Pipeline${NC}"
    python finetune_forecasting.py --train-from-scratch

elif [ "$1" == "--benchmark-eval" ]; then
    echo -e "${GREEN}Running Benchmark Evaluation (TSFM vs TimesFM)${NC}"
    python evaluate_checkpoints.py \
        --models both \
        --checkpoint-glob "tsfm_best.pt" \
        --forecast-horizon 96 \
        --datasets ETTh1 ETTh2 ETTm1 ETTm2 Electricity Traffic Weather

elif [ "$1" == "--eval" ]; then
    echo -e "${GREEN}Analyzing Fine-Tuning Results${NC}"
    python evaluate_results.py

elif [ "$1" == "--leaderboard" ]; then
    echo -e "${GREEN}Printing Benchmark Leaderboard${NC}"
    python benchmark_leaderboard.py \
        --input finetuning_results/benchmark_results.json \
        --sort-by mean_mse \
        --show-dataset-winners

elif [ "$1" == "--linear-probe-only" ]; then
    echo -e "${GREEN}Running Linear Probe Only (Fast Validation)${NC}"
    python finetune_forecasting.py --linear-probe-epochs 5 --finetune-epochs 0

elif [ "$1" == "--quick" ]; then
    echo -e "${GREEN}Quick Smoke Test (2 epochs, synthetic data)${NC}"
    echo ""
    echo -e "${BLUE}  Pretrain (1 epoch, 20 steps)...${NC}"
    python train_tsfm.py \
        --data-dir data \
        --epochs 1 \
        --max-steps-per-epoch 20 \
        --loss-fn huber \
        --no-augment \
        --ema-decay 0 \
        --best-model-path tsfm_best.pt
    echo ""
    echo -e "${BLUE}  Fine-tune (2 epochs)...${NC}"
    python finetune_forecasting.py \
        --linear-probe-epochs 2 \
        --finetune-epochs 2 \
        --batch-size 16

elif [ "$1" == "--download-data" ]; then
    echo -e "${GREEN}Downloading All Data${NC}"
    python download_pretraining_corpora.py \
        --output-dir data/real_corpora \
        --max-datasets 10
    python prepare_datasets.py --timesfm-benchmarks

elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: bash quickstart.sh [COMMAND]"
    echo ""
    echo "Pipeline Commands:"
    echo "  (no args)             Run complete pipeline (download → pretrain → ablation → fine-tune → eval)"
    echo "  --pretrain            Pretrain with Huber loss, EMA, augmentation, and early stopping"
    echo "  --pretrain-memory-safe Pretrain with gradient checkpointing (saves ~40% VRAM)"
    echo "  --pretrain-cpu        Pretrain on CPU (no AMP, no compile, smaller batch)"
    echo "  --ablation            Full 7-seed ablation with Welch's t-test + Holm-Bonferroni"
    echo "  --ablation-pilot      3-seed pilot + seed count estimation"
    echo "  --finetune            Run fine-tuning with from-scratch baseline"
    echo "  --benchmark-eval      Evaluate TSFM vs TimesFM on 7/7 benchmarks"
    echo "  --eval                Analyze fine-tuning results and generate plots"
    echo "  --leaderboard         Print ranked benchmark leaderboard"
    echo "  --linear-probe-only   Run linear probe validation (fast)"
    echo "  --quick               Quick smoke test (1 pretrain epoch + 2 finetune epochs)"
    echo "  --download-data       Download real corpora + benchmark datasets"
    echo "  --help                Show this help message"
    echo ""
    echo "New Flags Available:"
    echo "  --loss-fn {mse,huber}         Loss function (default: huber)"
    echo "  --ema-decay FLOAT             EMA weight averaging (default: 0.999, 0=disabled)"
    echo "  --augment / --no-augment      Time-series augmentation (default: on)"
    echo "  --gradient-checkpointing      Trade speed for ~40% memory savings"
    echo "  --no-compile                  Disable torch.compile optimization"
    echo ""
    echo "Examples:"
    echo "  bash quickstart.sh                     # Full pipeline"
    echo "  bash quickstart.sh --quick             # Fast smoke test"
    echo "  bash quickstart.sh --pretrain           # Just pretrain"
    echo "  bash quickstart.sh --ablation-pilot     # Quick statistical pilot"

else
    echo -e "${YELLOW}[ERROR] Unknown command: $1${NC}"
    echo "Run 'bash quickstart.sh --help' for usage information"
    exit 1
fi

echo ""
echo "=================================================="
echo "For detailed documentation, see: README.md"
echo "=================================================="
