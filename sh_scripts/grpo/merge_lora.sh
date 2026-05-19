#!/bin/bash
# =================================================
# Fuzja wag SFT LoRA z modelem bazowym Qwen3-4B
# =================================================
#SBATCH --job-name=merge_sft_lora
#SBATCH --output=Agents/out/merge_lora_%j.out
#SBATCH --time=0-00:20:00
#SBATCH -p lem-cpu-short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64gb

set -e

# 1. Ścieżki - BASE_DIR to główny folder projektu
MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"
VENV_PATH="$BASE_DIR/venv"

# Ścieżka do skryptu pythonowego (załóżmy, że zapiszesz merge_lora.py bezpośrednio w folderze Agents)
PYTHON_SCRIPT="$BASE_DIR/merge_lora.py"

# 2. Ładowanie modułów
source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0

# 3. Aktywacja środowiska
source "$VENV_PATH/bin/activate"

# Zabezpieczenie ścieżek cache dla HuggingFace (ochrona przed limitami na /home)
export MY_NEW_TMP="$MY_DISK/tmp"
export HF_HOME="$MY_NEW_TMP/hf_cache"
mkdir -p "$HF_HOME"

# 4. Uruchomienie skryptu
echo "================================================="
echo "=> Startowanie procesu fuzji wag (Merge LoRA)..."
echo "=> Skrypt Pythona: $PYTHON_SCRIPT"
echo "================================================="

python "$PYTHON_SCRIPT"

echo "================================================="
echo "=> Sukces! Adapter SFT został wtopiony w model bazowy."
echo "=> Nowy model czeka pod ścieżką zdefiniowaną w output_path."
echo "================================================="