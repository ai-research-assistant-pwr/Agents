#!/bin/bash
#SBATCH --job-name=data_subsetting
#SBATCH --output=Agents/out/test_df_%j.out
#SBATCH --time=0-00:10:00
#SBATCH -p lem-cpu-short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16gb

set -e

# 1. Ścieżki - BASE_DIR to główny folder projektu
MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"
VENV_PATH="$BASE_DIR/venv"

# Ścieżka do skryptu pythonowego zgodnie z Twoją strukturą
PYTHON_SCRIPT="$BASE_DIR/src/grpo/grpo_train/create_test_dfs.py"

# 2. Ładowanie modułów
source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

# 3. Aktywacja środowiska
source $VENV_PATH/bin/activate

# 4. Uruchomienie skryptu
echo "=> Startowanie procesu podziału danych..."
echo "=> Używam skryptu: $PYTHON_SCRIPT"

python "$PYTHON_SCRIPT"

echo "=> Sukces! Pliki CSV zostały wygenerowane w folderze danych."