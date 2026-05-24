#!/bin/bash
# =============================================================================
# run_attention_probe.sh — Ekstrakcja Map Uwagi (Generator Attention)
# =============================================================================
#SBATCH --job-name=attn_probe
#SBATCH --output=Agents/out/%x_%j.out
#SBATCH --time=0-01:00:00
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:hopper:1
#SBATCH --ntasks-per-node=1

set -e

MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"
VENV_PATH="$BASE_DIR/venv"
SCRIPT_PATH="$BASE_DIR/src/eval/extract_attention.py"

echo "========================================================"
echo " Uruchamianie Ekstrakcji Map Uwagi"
echo " Skrypt: $SCRIPT_PATH"
echo "========================================================"

source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0

source $VENV_PATH/bin/activate

# Instalacja seaborn w locie, jeśli brakuje
pip install seaborn matplotlib --quiet

# =============================================================================
# Zarządzanie Cache (Krytyczne na WCSS)
# =============================================================================
export MY_NEW_TMP="$MY_DISK/tmp"
export XDG_CACHE_HOME="$MY_NEW_TMP/xdg_cache"
export TRITON_CACHE_DIR="$MY_NEW_TMP/triton_cache"
export HF_HOME="$MY_NEW_TMP/hf_cache"
mkdir -p "$MY_NEW_TMP" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$HF_HOME"

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# Wymuszenie jednego GPU (dla bezpieczeństwa przy eager attention)
export CUDA_VISIBLE_DEVICES="0"

echo "=> Rozpoczynam inferencję z eager attention..."
python "$SCRIPT_PATH"

echo "========================================================"
echo " Gotowe! Poszukaj pliku .png w katalogu ewaluacyjnym."
echo "========================================================"