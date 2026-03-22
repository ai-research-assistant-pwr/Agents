#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32gb
#SBATCH --time=0-00:30:00
#SBATCH --job-name=dense_retrieval
#SBATCH --output=Agents/out/retrieval.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

source /home/tymrom7227/disk/venvs/pnw-2/bin/activate
VENV_PYTHON="/home/tymrom7227/disk/venvs/pnw-2/bin/python"

AGENTS_DIR="$HOME/disk/Agents"
export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

GET_CONFIG="$VENV_PYTHON $AGENTS_DIR/src/sft/utils/config.py"
MY_DISK=$($GET_CONFIG paths.base_path)

export HOME=$MY_DISK
export XDG_CACHE_HOME=$MY_DISK/.cache

echo "====================================="
echo "Starting Dense Retrieval..."
echo "====================================="

python3 -m pip install faiss-cpu
python3 $AGENTS_DIR/scripts/sft/run_retrieval.py
echo "Done!"