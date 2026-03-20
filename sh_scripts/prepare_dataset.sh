#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32gb
#SBATCH --time=0-00:30:00
#SBATCH --job-name=dense_retrieval
#SBATCH --output=Agents/out/prep_dataset_sft.out
#SBATCH -p lem-cpu

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

source ~/disk/venvs/pnw-2/bin/activate

AGENTS_DIR="$HOME/disk/Agents"
export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

echo "====================================="
echo " Preparing SFT Dataset..."
echo "====================================="

python $AGENTS_DIR/src/train/01_prepare_datasets.py

echo "Done!"