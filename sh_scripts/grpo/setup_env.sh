#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --time=0-01:00:00
#SBATCH --job-name=setup_marti
#SBATCH --output=setup_marti.out
#SBATCH -p lem-cpu

echo "=========================================="
echo "Starting setup for PNW-2"
echo "=========================================="

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

source /home/tymrom7227/disk/venvs/pnw-2/bin/activate

cd /home/tymrom7227/disk

echo "-> Cloning repository MARTI (TsinghuaC3I)..."
if [ ! -d "MARTI" ]; then
    git clone https://github.com/TsinghuaC3I/MARTI.git
else
    echo "Directory MARTI already exists, skipping clone."
fi

cd MARTI

echo "-> Installing MARTI (OpenRLHF) and vLLM (this may take a while)..."
# Installation in editable mode with dependencies for vLLM

echo "-> Installing additional utility libraries..."
pip install ray wandb pyyaml transformers peft

echo "=========================================="
echo "Installation completed successfully"
echo "=========================================="