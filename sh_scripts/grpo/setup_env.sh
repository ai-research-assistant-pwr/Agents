#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=0-01:30:00
#SBATCH --job-name=setup_marti
#SBATCH --output=/home/tymrom7227/disk/Agents/out/setup_marti.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

set -e

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

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



echo "-> Installing PyTorch 2.4.0 with CUDA 12.4 binaries..."
$VENV_PYTHON -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

echo "-> Managing MARTI repository..."
cd $MY_DISK
if [ ! -d "MARTI" ]; then
    echo "Cloning MARTI from GitHub..."
    git clone https://github.com/TsinghuaC3I/MARTI.git
else
    echo "Directory MARTI already exists, skipping clone."
fi

cd $MY_DISK/MARTI
echo "-> Installing MARTI and vLLM dependencies..."
$VENV_PYTHON -m pip install --no-build-isolation -e .[vllm]

echo "-> Installing utility libraries..."
$VENV_PYTHON -m pip install ray wandb pyyaml transformers peft

echo "=========================================="
echo "SETUP COMPLETED SUCCESSFULLY"
echo "=========================================="