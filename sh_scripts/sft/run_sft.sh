#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=0-04:00:00
#SBATCH --job-name=sft_qwen
#SBATCH --output=Agents/out/sft_qwen.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
source /home/tymrom7227/disk/venvs/pnw-2/bin/activate
VENV_PYTHON="/home/tymrom7227/disk/venvs/pnw-2/bin/python"

AGENTS_DIR="$HOME/disk/Agents"
export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

GET_CONFIG="$VENV_PYTHON $AGENTS_DIR/src/sft/utils/config.py"
MY_DISK=$($GET_CONFIG paths.base_path)

export XDG_CACHE_HOME=$MY_DISK/.cache
export HOME=$MY_DISK
export HF_HOME=$MY_DISK/.cache/hf
export TORCHINDUCTOR_CACHE_DIR=$MY_DISK/.cache/torch_inductor

export WANDB_MODE="offline" 
export TRANSFORMERS_OFFLINE=1 
export HF_DATASETS_OFFLINE=1

echo "================================================="
echo "TESTING CONFIGURATION BEFORE SFT TRAINING"
echo "================================================="

DIRS_TO_CHECK=(
    "$AGENTS_DIR"
    "$AGENTS_DIR/data/datasets"
    "$MY_DISK/models"
)

for DIR in "${DIRS_TO_CHECK[@]}"; do
    if [ ! -d "$DIR" ]; then
        echo "CRITICAL ERROR: Directory not found: $DIR"
        exit 1
    else
        echo "Directory exists: $DIR"
    fi
done

FILES_TO_CHECK=(
    "$AGENTS_DIR/src/sft/sft_train/run_sft.py"
    "$AGENTS_DIR/data/datasets/retriever_train.jsonl"
    "$AGENTS_DIR/data/datasets/generator_train.jsonl"
)

for FILE in "${FILES_TO_CHECK[@]}"; do
    if [ ! -f "$FILE" ]; then
        echo "CRITICAL ERROR: File not found: $FILE"
        exit 1
    else
        echo "File exists: $FILE"
    fi
done

echo "================================================="
echo "All paths verified successfully. Starting SFT training..."
echo "================================================="


$VENV_PYTHON $AGENTS_DIR/src/sft/sft_train/run_sft.py --task retriever
$VENV_PYTHON $AGENTS_DIR/src/sft/sft_train/run_sft.py --task generator

echo "====================================="
echo "Training completed successfully!"
echo "====================================="