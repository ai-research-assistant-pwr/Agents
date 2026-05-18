#!/bin/bash

#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=8gb
#SBATCH --time=0-00:10:00
#SBATCH --job-name=prep_sft
#SBATCH --output=Agents/out/prep_sft_%j.out
#SBATCH -p lem-cpu

set -e 

MY_DISK="$SLURM_SUBMIT_DIR"
AGENTS_DIR="$MY_DISK/Agents"
VENV_PATH="$AGENTS_DIR/venv"

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

source "$VENV_PATH/bin/activate"
VENV_PYTHON="$VENV_PATH/bin/python"

export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

export XDG_CACHE_HOME="$MY_DISK/.cache"
export HF_HOME="$MY_DISK/.cache/hf"

echo "================================================="
echo "Configuration Loaded:"
echo "  Disk Root Path: $MY_DISK"
echo "  Agents Path: $AGENTS_DIR"
echo "  Virtual Env: $VENV_PATH"
echo "================================================="

echo "Starting SFT Dataset Preparation..."

set +e
$VENV_PYTHON "$AGENTS_DIR/src/sft/sft_train/prepare_dataset.py"
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
    echo "====================================="
    echo "Dataset Preparation completed SUCCESSFULLY."
    echo "====================================="
else
    echo "====================================="
    echo "Dataset Preparation FAILED (exit code $EXIT_CODE)."
    echo "====================================="
    exit $EXIT_CODE
fi