#!/bin/bash
#SBATCH --job-name=data_subsetting
#SBATCH --output=Agents/out/test_df.out
#SBATCH --time=0-00:10:00        # 10 minut to aż nadto
#SBATCH -p lem-cpu-short         # Używamy partycji CPU (szybciej wystartuje)
#SBATCH -N 1
#SBATCH -c 4                     # 4 rdzenie wystarczą do Pandasa
#SBATCH --mem=16gb

set -e

# 1. Ścieżki
MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"
VENV_PATH="$BASE_DIR/venv"

# 2. Ładowanie modułów (tak samo jak w Twoim głównym skrypcie)
source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

# 3. Aktywacja środowiska
source $VENV_PATH/bin/activate

# 4. Uruchomienie skryptu Pythona
echo "=> Startowanie procesu podziału danych..."
python "$BASE_DIR/create_test_dfs.py"

echo "=> Sukces! Pliki CSV zostały wygenerowane."