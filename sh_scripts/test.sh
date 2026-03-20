#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32gb
#SBATCH --time=0-00:30:00
#SBATCH --job-name=dense_retrieval
#SBATCH --output=Agents/out/test.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

source ~/disk/venvs/pnw-2/bin/activate

MY_DISK="$HOME/disk"
AGENTS_DIR="$MY_DISK/Agents"

export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

VENV_PYTHON="$MY_DISK/venvs/pnw-2/bin/python"

GET_CONFIG="$VENV_PYTHON $AGENTS_DIR/src/utils/config.py"

echo "======================================================"
echo "      PEŁNA WERYFIKACJA PROCESU Z CONFIG.YAML"
echo "======================================================"

echo ""
echo "--- 1. KATALOGI GŁÓWNE ---"
echo "Dysk bazowy (base_path):          $($GET_CONFIG paths.base_path)"
echo "Katalog Agents:                   $($GET_CONFIG paths.base_path_agents)"
echo "Katalog artykułów (articles_dir): $($GET_CONFIG paths.articles_dir)"
echo "Katalog wektorów (embeddings_dir):$($GET_CONFIG paths.embeddings_dir)"

echo ""
echo "--- 2. UŻYWANE MODELE ---"
echo "Model generujący (LLM):           $($GET_CONFIG models.generation_model)"
echo "Model wektorujący (Embeddings):   $($GET_CONFIG models.embedding_model)"

echo ""
echo "--- 3. BAZA WIEDZY (RAG) ---"
echo "Katalog z indeksami wektorów:     $($GET_CONFIG files.index)"

echo ""
echo "--- 4. PLIKI W POTOKU (PIPELINE) ---"
echo "ETAP 1 (Wejście) - Prompty:       $($GET_CONFIG files.prompts)"
echo "ETAP 2 (Wyjście) - Wektory:       $($GET_CONFIG files.prompt_embeddings)"
echo "ETAP 3 (Wyjście) - Konteksty RAG: $($GET_CONFIG files.retrieved_contexts)"
echo "ETAP 4 (Wyjście) - Zbiór SFT:     $($GET_CONFIG files.synthetic_sft_dataset)"