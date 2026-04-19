#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --time=0-01:00:00
#SBATCH --job-name=setup_marti
#SBATCH --output=/home/tymrom7227/disk/Agents/out/setup_marti.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

set -e

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.4.1

source /home/tymrom7227/disk/venvs/pnw-2/bin/activate
VENV_PYTHON="/home/tymrom7227/disk/venvs/pnw-2/bin/python"

MY_DISK="/home/tymrom7227/disk"
AGENTS_DIR="$MY_DISK/Agents"

export XDG_CACHE_HOME=$MY_DISK/.cache
export HOME=$MY_DISK
export HF_HOME=$MY_DISK/.cache/hf
export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

echo "=========================================="
echo "START SETUP"
echo "=========================================="

$VENV_PYTHON -m pip install --upgrade pip setuptools wheel packaging

$VENV_PYTHON -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

cd $MY_DISK/MARTI
echo "-> Installing MARTI with vLLM (using no-build-isolation)..."
$VENV_PYTHON -m pip install --no-build-isolation -e .[vllm]

# 4. Instalacja pozostałych bibliotek
echo "-> Installing utility libraries..."
$VENV_PYTHON -m pip install ray wandb pyyaml transformers peft

echo "=========================================="
echo "SETUP COMPLETED: $(date)"
echo "=========================================="