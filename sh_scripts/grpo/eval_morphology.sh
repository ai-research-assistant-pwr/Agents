#!/bin/bash
# =================================================
# Evaluation Job: Experiment 1 (Morphology)
# =================================================
#SBATCH --job-name=eval_exp1
#SBATCH --output=Agents/out/eval_morphology.out
#SBATCH --time=0-00:30:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16gb

set -e 

MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"
VENV_PATH="$BASE_DIR/venv"

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

source $VENV_PATH/bin/activate
VENV_PYTHON="$VENV_PATH/bin/python"

echo "=> Job executing on node: $(hostname)"
echo "=> Base Dir resolved to: $BASE_DIR"
echo "=> Starting Experiment 1 Morphology Analysis..."

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

$VENV_PYTHON "$BASE_DIR/src/eval/eval_morphology.py"

echo "=> Evaluation completed successfully! Check Agents/eval_results/experiment_1 for outputs."