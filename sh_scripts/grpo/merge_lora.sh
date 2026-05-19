#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=0-00:20:00
#SBATCH --job-name=merge_lora
#SBATCH --output=Agents/out/%x_%j.out
#SBATCH -p lem-cpu-short

# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   sbatch merge_lora.sh
# ─────────────────────────────────────────────────────────────────────────────

MY_DISK="$SLURM_SUBMIT_DIR"
AGENTS_DIR="$MY_DISK/Agents"
VENV_PATH="$AGENTS_DIR/venv"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "CRITICAL ERROR: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0

source "$VENV_PATH/bin/activate"
VENV_PYTHON="$VENV_PATH/bin/python"

export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

export MY_NEW_TMP="$MY_DISK/tmp"
export XDG_CACHE_HOME="$MY_NEW_TMP/xdg_cache"
export HF_HOME="$MY_NEW_TMP/hf_cache"
export TRITON_CACHE_DIR="$MY_NEW_TMP/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$MY_NEW_TMP/torchinductor_cache"
mkdir -p "$MY_NEW_TMP" "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "================================================="
echo "Starting LoRA Merge (SFT -> Base Model) ..."
echo "Executing: $PYTHON_SCRIPT"
echo "================================================="

$VENV_PYTHON "$PYTHON_SCRIPT"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "====================================="
    echo "LoRA Merge completed successfully."
    echo "====================================="
else
    echo "====================================="
    echo "LoRA Merge FAILED (exit code $EXIT_CODE)."
    echo "====================================="
    exit $EXIT_CODE
fi