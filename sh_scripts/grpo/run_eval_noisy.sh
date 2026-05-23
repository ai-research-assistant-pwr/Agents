#!/bin/bash
#SBATCH --job-name=eval_noisy
#SBATCH --output=Agents/out/%x_%j.out
#SBATCH --time=0-04:00:00
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:hopper:1
#SBATCH --ntasks-per-node=1

set -e

# =============================================================================
# Konfiguracja ścieżek i parametrów
# =============================================================================
MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"
VENV_PATH="$BASE_DIR/venv"

# TODO: Ustaw ścieżkę do wyuczonego modelu SFT/GRPO
TRAIN_MODEL="${TRAIN_MODEL:-"$BASE_DIR/models_output/grpo_qwen_5231830_results"}"
EMBED_MODEL="${EMBED_MODEL:-"Qwen/Qwen3-Embedding-4B"}"

EVAL_DATA_PATH="$BASE_DIR/data/datasets/rl_grounded_dataset_eval.csv"
WORKFLOW_SCRIPT="$BASE_DIR/src/grpo/grpo_train/scientific_workflow.py"
DEBUG_DIR="$BASE_DIR/workflow_logs/eval_noisy"

EMBED_PORT=8000

# =============================================================================
# Inicjalizacja środowiska
# =============================================================================
source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0

source "$VENV_PATH/bin/activate"
VENV_PYTHON="$VENV_PATH/bin/python"

export MY_NEW_TMP="$MY_DISK/tmp"
export XDG_CACHE_HOME="$MY_NEW_TMP/xdg_cache"
export TRITON_CACHE_DIR="$MY_NEW_TMP/triton_cache"
mkdir -p "$MY_NEW_TMP" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$DEBUG_DIR"

# =============================================================================
# Start serwera vLLM dla embeddingów (w tle)
# =============================================================================
echo "=> Uruchamianie serwera vLLM (Embeddings) na porcie $EMBED_PORT..."
$VENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL" \
    --host 0.0.0.0 \
    --port $EMBED_PORT \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.3 &
VLLM_PID=$!

echo "=> Oczekiwanie na gotowość serwera vLLM..."
while ! curl -s http://localhost:$EMBED_PORT/v1/models > /dev/null; do
    sleep 5
done
echo "=> Serwer vLLM jest online!"

# =============================================================================
# Konfiguracja Agenta i uruchomienie inferencji
# =============================================================================
AGENT0="{
    \"0\": {
        \"agent_id\": \"shared_agent\",
        \"role\": \"generator\",
        \"agent_role\": \"generator\",
        \"pretrain\": \"$TRAIN_MODEL\",
        \"is_tuning\": false,
        \"is_reasoning_model\": true
    }
}"

echo "=> Rozpoczęcie ewaluacji z zamaskowanym kanałem komunikacyjnym (Intervention)..."

$VENV_PYTHON -m marti.cli.multi_agent_train_ppo_ray \
    --pretrain "$TRAIN_MODEL" \
    --agents "$AGENT0" \
    --workflow_func_path "$WORKFLOW_SCRIPT" \
    --eval_dataset "$EVAL_DATA_PATH" \
    --eval_split test \
    --eval_only \
    --eval_n_samples_per_prompt 10 \
    --workflow_args "{\"debug_dir\": \"$DEBUG_DIR\", \"apply_channel_noise\": \"true\", \"noise_probability\": 1.0, \"embed_host\": \"localhost\", \"embed_port\": $EMBED_PORT}" \
    --input_key "user_query" \
    --label_key "hypothesis" \
    --metadata_key "metadata" \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.6 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1

echo "=> Ewaluacja (Noisy) zakończona pomyślnie!"

# Sprzątanie
kill "$VLLM_PID" 2>/dev/null || true
echo "=> Serwer vLLM wyłączony."