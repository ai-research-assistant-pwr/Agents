#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=0-01:00:00
#SBATCH --job-name=sim_pipeline
#SBATCH --output=Agents/out/sim_pipeline.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
source /home/tymrom7227/disk/venvs/pnw-2/bin/activate
VENV_PYTHON="/home/tymrom7227/disk/venvs/pnw-2/bin/python"

MY_DISK="/home/tymrom7227/disk"
AGENTS_DIR="$MY_DISK/Agents"

export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

export XDG_CACHE_HOME=$MY_DISK/.cache
export HF_HOME=$MY_DISK/.cache/hf
export TORCHINDUCTOR_CACHE_DIR=$MY_DISK/.cache/torch_inductor
export TRANSFORMERS_OFFLINE=0

echo "================================================="
echo "TESTING CONFIGURATION BEFORE SIMULATION"
echo "================================================="

MODELS_OUT_DIR="$MY_DISK/models_output/run3"

DIRS_TO_CHECK=(
    "$AGENTS_DIR"
    "$AGENTS_DIR/data/datasets/sft"
    "$MY_DISK/models"
    "$MODELS_OUT_DIR"
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
    "$AGENTS_DIR/src/sft/sft_train/simulate_pipeline.py"
    "$AGENTS_DIR/data/datasets/sft/retriever_test.jsonl"
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
echo "All paths verified successfully. Starting Simulation..."
echo "================================================="

echo ">>> RUNNING END-TO-END PIPELINE SIMULATION <<<"

$VENV_PYTHON $AGENTS_DIR/src/sft/sft_train/simulate_pipeline.py

echo "====================================="
echo "Simulation completed successfully!"
echo "====================================="