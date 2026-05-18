#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=0-04:00:00
#SBATCH --job-name=sft_shared
#SBATCH --output=Agents/out/%x_%j.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   sbatch run_sft.sh YOUR_WANDB_API_KEY
# ─────────────────────────────────────────────────────────────────────────────

WANDB_API_KEY=$1

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: No WANDB API key provided."
    echo "Usage: sbatch run_sft.sh YOUR_SECRET_API_KEY"
    exit 1
fi

MY_DISK="$SLURM_SUBMIT_DIR"
AGENTS_DIR="$MY_DISK/Agents"
VENV_PATH="$AGENTS_DIR/venv"

source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0

source "$VENV_PATH/bin/activate"
VENV_PYTHON="$VENV_PATH/bin/python"

$VENV_PYTHON -m pip install trl peft datasets pandas

export WANDB_DIR="$AGENTS_DIR/wandb"
mkdir -p "$WANDB_DIR"

export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

export MY_NEW_TMP="$MY_DISK/tmp"
export XDG_CACHE_HOME="$MY_NEW_TMP/xdg_cache"
export HF_HOME="$MY_NEW_TMP/hf_cache"
export TRITON_CACHE_DIR="$MY_NEW_TMP/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$MY_NEW_TMP/torchinductor_cache"
mkdir -p "$MY_NEW_TMP" "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_MODE="online"
export TRANSFORMERS_OFFLINE=0

echo "================================================="
echo "VERIFYING PATHS FOR SHARED AGENT TRAINING"
echo "================================================="

# Dataset directory
SFTMOD_DIR="$AGENTS_DIR/data/datasets/sft_mod"
TRAIN_DATASET="$SFTMOD_DIR/shared_agent_train.jsonl"
EVAL_DATASET="$SFTMOD_DIR/shared_agent_eval.jsonl"

if [ ! -f "$TRAIN_DATASET" ]; then
    echo "CRITICAL ERROR: Training dataset not found at $TRAIN_DATASET"
    echo ""
    echo "Run prepare_dataset.py first:"
    echo "  python $AGENTS_DIR/src/sft/sft_train/prepare_dataset.py"
    echo ""
    echo "Input CSVs expected at:"
    echo "  $AGENTS_DIR/data/datasets/rl_retriever_outputs.csv"
    echo "  $AGENTS_DIR/data/datasets/rl_generator_outputs_no_mask.csv"
    echo "  $AGENTS_DIR/data/datasets/rl_generator_outputs_mask.csv  (optional)"
    exit 1
fi

echo "Train dataset:  $TRAIN_DATASET  [FOUND]"

if [ -f "$EVAL_DATASET" ]; then
    echo "Eval dataset:   $EVAL_DATASET  [FOUND]"
else
    echo "Eval dataset:   $EVAL_DATASET  [NOT FOUND — training without eval]"
fi

echo "================================================="
echo "Starting UNIFIED SFT training (Shared LoRA) ..."
echo "================================================="

$VENV_PYTHON "$AGENTS_DIR/src/sft/sft_train/run_sft.py" --task shared_agent

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "====================================="
    echo "SFT Training completed successfully."
    echo "====================================="
else
    echo "====================================="
    echo "SFT Training FAILED (exit code $EXIT_CODE)."
    echo "====================================="
    exit $EXIT_CODE
fi