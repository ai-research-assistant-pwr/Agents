#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32gb
#SBATCH --time=0-00:30:00
#SBATCH --job-name=dense_retrieval
#SBATCH --output=Agents/out/retrieval.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

source ~/disk/venvs/pnw-2/bin/activate

MY_DISK="$HOME/disk"
AGENTS_DIR="$MY_DISK/Agents"

export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

export HOME=$MY_DISK
export XDG_CACHE_HOME=$MY_DISK/.cache

echo "====================================="
echo "Starting Dense Retrieval..."
echo "====================================="

python3 -m pip install faiss-cpu
python3 $AGENTS_DIR/src/run_retrieval.py
echo "Done!"