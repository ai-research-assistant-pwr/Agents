#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --time=0-01:00:00
#SBATCH --job-name=setup_marti
#SBATCH --output=/home/tymrom7227/disk/Agents/out/setup_marti.out
#SBATCH -p lem-cpu-short

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
echo "Starting setup for PNW-2 on $MY_DISK"
echo "=========================================="

cd $MY_DISK

if [ ! -d "MARTI" ]; then
    echo "-> Cloning MARTI..."
    git clone https://github.com/TsinghuaC3I/MARTI.git
fi

cd MARTI

echo "-> Installing MARTI and vLLM..."
$VENV_PYTHON -m pip install --upgrade pip
$VENV_PYTHON -m pip install -e .[vllm]

echo "-> Installing additional utility libraries..."
$VENV_PYTHON -m pip install ray wandb pyyaml transformers peft

echo "=========================================="
echo "Installation completed successfully"
echo "=========================================="