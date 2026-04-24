#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=0-04:00:00
#SBATCH --job-name=sft_shared
#SBATCH --output=Agents/out/sft_shared.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

WANDB_API_KEY=$1

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: No WANDB API key provided."
    echo "Usage: sbatch run_sft.sh YOUR_SECRET_API_KEY"
    exit 1
fi

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
source /home/tymrom7227/disk/venvs/pnw-2/bin/activate
VENV_PYTHON="/home/tymrom7227/disk/venvs/pnw-2/bin/python"

echo "================================================="
echo "APPLYING TEMPORARY ENVIRONMENT FIXES (pnw-2)"
echo "================================================="
# 1. Wymuszenie instalacji nowszego PyTorcha (kompatybilnego z cu124 i torchao)
$VENV_PYTHON -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Naprawa uszkodzonych zależności (transformers) i aktualizacja peft/trl
$VENV_PYTHON -m pip install --upgrade transformers peft trl accelerate torchao
echo "================================================="

MY_DISK="/home/tymrom7227/disk"
AGENTS_DIR="$MY_DISK/Agents"

export WANDB_DIR="$AGENTS_DIR/wandb"
mkdir -p "$WANDB_DIR"

export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

GET_CONFIG="$VENV_PYTHON $AGENTS_DIR/src/sft/utils/config.py"
MY_DISK=$($GET_CONFIG paths.base_path)

export XDG_CACHE_HOME=$MY_DISK/.cache
export HF_HOME=$MY_DISK/.cache/hf
export TORCHINDUCTOR_CACHE_DIR=$MY_DISK/.cache/torch_inductor

export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_MODE="online"
export TRANSFORMERS_OFFLINE=0

echo "================================================="
echo "VERIFYING PATHS FOR SHARED AGENT TRAINING"
echo "================================================="

SHARED_DATASET="$AGENTS_DIR/data/datasets/sft3/shared_agent_train.jsonl"

if [ ! -f "$SHARED_DATASET" ]; then
    echo "CRITICAL ERROR: Shared dataset not found at $SHARED_DATASET"
    echo "Please run 'python prepare_dataset.py' first."
    exit 1
fi

echo "================================================="
echo "Starting UNIFIED SFT training (Shared Model)..."
echo "================================================="

$VENV_PYTHON $AGENTS_DIR/src/sft/sft_train/run_sft.py --task shared_agent

echo "====================================="
echo "Unified SFT Training completed successfully!"
echo "====================================="