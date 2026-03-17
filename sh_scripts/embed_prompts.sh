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

source ~/disk/venvs/pnw-2/bin/activate 

GET_CONFIG="python3 Agents/src/utils/config.py"

echo "====================================="
echo "🔍 PATH DEBUGGER START"
echo "====================================="
echo "1. Obecna lokalizacja (pwd): $(pwd)"
echo "2. Użytkownik: $(whoami)"
echo "3. Czy venv python istnieje? $(ls -l $VENV_PYTHON 2>/dev/null || echo 'NIE ZNALEZIONO!')"

# Próba wyciągnięcia ścieżek z configu
MY_DISK=$($GET_CONFIG paths.base_path)
EMBED_MODEL_CFG=$($GET_CONFIG models.embedding_model)

echo "4. MY_DISK z configu: $MY_DISK"
echo "5. Ścieżka modelu z configu: $EMBED_MODEL_CFG"

echo "6. Sprawdzam czy model fizycznie tam jest:"
if [ -d "$EMBED_MODEL_CFG" ]; then
    echo "   ✅ FOLDER MODELU ISTNIEJE"
    echo "   📄 Zawartość folderu (szukamy config.json):"
    ls -F "$EMBED_MODEL_CFG" | head -n 5
else
    echo "   ❌ BŁĄD: Folder modelu NIE ISTNIEJE pod tą ścieżką!"
    echo "   🔍 Szukam gdziekolwiek folderu 'snapshots' w models:"
    find ~/disk/models -name "snapshots" -type d 2>/dev/null | head -n 3
fi

echo "7. Sprawdzam plik z promptami:"
PROMPTS_FILE="$MY_DISK/data/prompts/generated_prompts_gemini.csv"
if [ -f "$PROMPTS_FILE" ]; then
    echo "   ✅ PLIK PROMPTÓW ISTNIEJE ($PROMPTS_FILE)"
else
    echo "   ❌ BŁĄD: Plik promptów NIE ISTNIEJE w $PROMPTS_FILE"
fi
echo "====================================="

export TORCHINDUCTOR_CACHE_DIR=$MY_DISK/.cache/torch_inductor
export VLLM_CACHE_ROOT=$MY_DISK/.cache/vllm
export HF_HOME=$MY_DISK/.cache/hf
export TRITON_CACHE_DIR=$MY_DISK/.cache/triton
export FLASHINFER_CACHE_DIR=$MY_DISK/.cache/flashinfer
export TORCH_EXTENSIONS_DIR=$MY_DISK/.cache/torch_extensions
export XDG_CACHE_HOME=$MY_DISK/.cache

export HF_HUB_OFFLINE=1

echo "Configuration Loaded:"
echo "  Agents Root Path: $MY_DISK"
echo "  VLLM Compile Cache: $VLLM_CACHE_ROOT"
echo "  Prompts Path: $PROMPTS_DIR"

rm -rf "$VLLM_CACHE_ROOT"
mkdir -p "$VLLM_CACHE_ROOT"
mkdir -p "$PROMPTS_DIR"

echo "====================================="
echo "Starting Prompts Embedding..."
echo "====================================="

~/disk/venvs/pnw-2/bin/python Agents/src/embed_prompts.py
echo "Done!"