#!/bin/bash

source ~/disk/venvs/pnw-2/bin/activate

MY_DISK="$HOME/disk"
AGENTS_DIR="$MY_DISK/Agents"

export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

VENV_PYTHON="$MY_DISK/venvs/pnw-2/bin/python"

GET_CONFIG="$VENV_PYTHON $AGENTS_DIR/src/utils/config.py"

echo "======================================================"
echo "      WERYFIKACJA ŚCIEŻEK Z CONFIG.YAML"
echo "======================================================"

echo ""
echo "--- KATALOGI BAZOWE ---"
echo "Dysk:           $($GET_CONFIG paths.base_path)"
echo "Agents:         $($GET_CONFIG paths.base_path_agents)"

echo ""
echo "--- NAJWAŻNIEJSZE PLIKI W PROCESIE GENEROWANIA ---"
echo "1. Prompty syntetyczne (Wejście):  $($GET_CONFIG files.prompts)"
echo "2. Wektory promptów (Wyjście):     $($GET_CONFIG files.prompt_embeddings)"
echo "3. Baza wektorowa (Index RAG):     $($GET_CONFIG files.index)"
echo "4. Pobrane konteksty (Wyjście):    $($GET_CONFIG files.retrieved_contexts)"
echo "5. Zbiór SFT (Końcowe wyjście):    $($GET_CONFIG files.synthetic_sft_dataset)"
echo "6. Hipotezy (Opcjonalne wyjście):  $($GET_CONFIG files.hypotheses)"

echo ""
echo "======================================================"
echo "Jeśli w plikach 1, 2, 4, 5 i 6 widzisz na końcu '_v2',"
echo "możesz bezpiecznie odpalać sbatch embed_prompts.sh!"
echo "======================================================"