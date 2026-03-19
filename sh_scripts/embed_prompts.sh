#!/bin/bash

#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=64gb
#SBATCH --time=0-01:00:00
#SBATCH --job-name=embed_prompts
#SBATCH --output=Agents/out/embed_prompts.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

source /home/tymrom7227/disk/venvs/pnw-2/bin/activate

VENV_PYTHON="/home/tymrom7227/disk/venvs/pnw-2/bin/python"
GET_CONFIG="$VENV_PYTHON Agents/src/utils/config.py"

MY_DISK=$($GET_CONFIG paths.base_path)
AGENTS_DIR=$($GET_CONFIG paths.base_path_agents)

export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

PROMPTS_DIR=$AGENTS_DIR/$($GET_CONFIG paths.prompts_dir_agents)
EMBEDDINGS_DIR=$MY_DISK/$($GET_CONFIG paths.embeddings_dir)

export XDG_CACHE_HOME=$MY_DISK/.cache
export HOME=$MY_DISK

export TORCHINDUCTOR_CACHE_DIR=$MY_DISK/.cache/torch_inductor
export VLLM_CACHE_ROOT=$MY_DISK/.cache/vllm
export HF_HOME=$MY_DISK/.cache/hf
export TRITON_CACHE_DIR=$MY_DISK/.cache/triton
export FLASHINFER_CACHE_DIR=$MY_DISK/.cache/flashinfer
export TORCH_EXTENSIONS_DIR=$MY_DISK/.cache/torch_extensions

echo "Configuration Loaded:"
echo "  Disk Root Path: $MY_DISK"
echo "  Agents Path: $AGENTS_DIR"
echo "  VLLM Compile Cache: $VLLM_CACHE_ROOT"
echo "  Prompts Path: $PROMPTS_DIR"

rm -rf "$VLLM_CACHE_ROOT"
mkdir -p $MY_DISK/.cache/vllm
mkdir -p $MY_DISK/.cache/torch_inductor

echo "====================================="
echo "Starting Prompts Embedding..."
echo "====================================="

# $VENV_PYTHON $AGENTS_DIR/src/embed_prompts.py

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

echo "Done!"