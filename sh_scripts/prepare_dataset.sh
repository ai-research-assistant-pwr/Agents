#!/bin/bash

#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=8gb
#SBATCH --time=0-00:10:00
#SBATCH --job-name=prep_sft
#SBATCH --output=Agents/out/prep_sft.out
#SBATCH -p lem-cpu

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

source /home/tymrom7227/disk/venvs/pnw-2/bin/activate

VENV_PYTHON="/home/tymrom7227/disk/venvs/pnw-2/bin/python"
GET_CONFIG="$VENV_PYTHON Agents/src/utils/config.py"

MY_DISK=$($GET_CONFIG paths.base_path)
AGENTS_DIR=$($GET_CONFIG paths.base_path_agents)

export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

export XDG_CACHE_HOME=$MY_DISK/.cache
export HOME=$MY_DISK
export HF_HOME=$MY_DISK/.cache/hf

echo "Configuration Loaded:"
echo "  Disk Root Path: $MY_DISK"
echo "  Agents Path: $AGENTS_DIR"

echo "====================================="
echo "Starting SFT Dataset Preparation..."
echo "====================================="

# $VENV_PYTHON $AGENTS_DIR/src/train/prepare_dataset.py

echo "Done!"