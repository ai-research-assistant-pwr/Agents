#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32gb
#SBATCH --time=0-00:30:00
#SBATCH --job-name=dense_retrieval
#SBATCH --output=Agents/out/test.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1


WANDB_API_KEY=$1
RELATIVE_LOG_PATH=$2

if [ -z "$WANDB_API_KEY" ] || [ -z "$RELATIVE_LOG_PATH" ]; then
    echo "Error: Missing arguments."
    echo "Usage: bash sync_manual_wandb.sh YOUR_API_KEY path/to/log"
    echo "Example: bash sync_manual_wandb.sh [API_KEY] disk/wandb/offline-run-20260322_..."
    exit 1
fi

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
source /home/tymrom7227/disk/venvs/pnw-2/bin/activate

export WANDB_API_KEY=$WANDB_API_KEY

FULL_PATH="$HOME/$RELATIVE_LOG_PATH"

if [ ! -d "$FULL_PATH" ]; then
    echo "Error: Directory does not exist: $FULL_PATH"
    echo "Make sure the path is correct (e.g., starts with 'disk/wandb/...')"
    exit 1
fi

echo "Synchronizing: $FULL_PATH"
echo "-------------------------------------------------"

python -m wandb sync "$FULL_PATH"

echo "-------------------------------------------------"
echo "Done!"