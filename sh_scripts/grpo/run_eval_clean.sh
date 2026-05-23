#!/bin/bash
# =============================================================================
# run_eval_clean.sh — Noise Probe: Clean Condition (noise_probability=0.0)
# =============================================================================
#
# Przeprowadza jeden przebieg eval-only na zbiorze rl_grounded_dataset_eval
# z noise_probability=0.0 i n_samples_per_prompt=10.
# Trajektorie zapisuje do workflow_logs/<PROBE_NAME>/eval_clean/eval/
#
# Uruchamiaj razem z run_eval_noisy.sh (mogą chodzić równolegle jako osobne joby).
#
# Usage:
#   sbatch sh_scripts/grpo/eval/run_eval_clean.sh <WANDB_API_KEY> <WEAVIATE_NODE_ID>
#
# Env overrides:
#   TRAIN_MODEL      — ścieżka do modelu (default: merged_sft_qwen_4B)
#   PROBE_NAME       — wspólna nazwa folderu z run_eval_noisy.sh (default: noise_probe)
#   EVAL_DATA_PATH   — zbiór eval (default: rl_grounded_dataset_eval.csv)
#   N_SAMPLES        — liczba próbek per prompt (default: 10)
#
# =============================================================================
#SBATCH --job-name=eval_clean
#SBATCH --output=Agents/out/%x_%j.out
#SBATCH --time=0-04:00:00
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=256gb
#SBATCH --gres=gpu:hopper:4
#SBATCH --ntasks-per-node=1

#SBATCH hetjob

# =================================================
# HET GROUP 1: Embedding Node (1 GPU)
# =================================================
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:hopper:1
#SBATCH --ntasks-per-node=2

set -e

# =============================================================================
# 1. Argumenty i konfiguracja
# =============================================================================
WANDB_API_KEY=$1
WEAVIATE_NODE_ID=$2

MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"
VENV_PATH="$BASE_DIR/venv"

TRAIN_MODEL="${TRAIN_MODEL:-"$BASE_DIR/models_output/grpo_qwen_5231830_results"}"
EMBED_MODEL="${EMBED_MODEL:-"Qwen/Qwen3-Embedding-4B"}"
RERANK_MODEL="${RERANK_MODEL:-"Qwen/Qwen3-Reranker-0.6B"}"

# Wspólna nazwa z run_eval_noisy.sh — muszą być takie same, żeby eval_noise_probe.py
# wiedział gdzie szukać obu katalogów
PROBE_NAME="${PROBE_NAME:-"noise_probe"}"

N_SAMPLES="${N_SAMPLES:-10}"

SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT:-1.2}"
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-0.3}"
GROUNDEDNESS_WEIGHT="${GROUNDEDNESS_WEIGHT:-1.0}"
RELEVANCY_WEIGHT="${RELEVANCY_WEIGHT:-1.0}"
USE_WEAVIATE_CONTEXT="${USE_WEAVIATE_CONTEXT:-false}"
WEAVIATE_TOP_N="${WEAVIATE_TOP_N:-5}"
ASK_RETRIEVER_LIMIT="${ASK_RETRIEVER_LIMIT:-0}"
RETRIEVER_SEARCH_LIMIT="${RETRIEVER_SEARCH_LIMIT:-0}"

# Szum: warunek CLEAN — wyłączamy flagę apply_channel_noise całkowicie
# żeby mieć pewność że żaden szum nie przesącza się przez is_eval=false check
NOISE_PROBABILITY="0.0"
APPLY_CHANNEL_NOISE="false"

EVAL_DATA_PATH="${EVAL_DATA_PATH:-"$BASE_DIR/data/datasets/rl_grounded_dataset_eval.csv"}"
WORKFLOW_SCRIPT="$BASE_DIR/src/grpo/grpo_train/scientific_workflow.py"

# Trajektorie trafią do eval/ (scientific_workflow.py dopisuje /eval gdy is_eval=true)
# Pełna ścieżka: workflow_logs/<PROBE_NAME>/eval_clean/eval/traj_*.json
DEBUG_DIR="$BASE_DIR/workflow_logs/$PROBE_NAME/eval_clean"
OUTPUT_DIR="$BASE_DIR/models_output/${PROBE_NAME}_clean_results"

EMBED_PORT=8000
RERANK_PORT=8001

# =============================================================================
# 2. Środowisko
# =============================================================================
source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0
source $VENV_PATH/bin/activate
VENV_PYTHON="$VENV_PATH/bin/python"

export MY_NEW_TMP="$MY_DISK/tmp"
export XDG_CACHE_HOME="$MY_NEW_TMP/xdg_cache"
export TRITON_CACHE_DIR="$MY_NEW_TMP/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$MY_NEW_TMP/torchinductor_cache"
export VLLM_USE_V1="0"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

mkdir -p "$MY_NEW_TMP" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$OUTPUT_DIR" "$DEBUG_DIR"

# =============================================================================
# 3. Node discovery
# =============================================================================
TRAIN_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0 | head -n 1)
EMBED_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1 | head -n 1)

echo "========================================================"
echo " Noise Probe — CLEAN condition"
echo " Probe name       : $PROBE_NAME"
echo " noise_probability: $NOISE_PROBABILITY"
echo " n_samples        : $N_SAMPLES"
echo " Train model      : $TRAIN_MODEL"
echo " Trajectories →   : $DEBUG_DIR/eval/"
echo " Trainer Node     : $TRAIN_NODE"
echo " Embed Node       : $EMBED_NODE"
echo "========================================================"

# =============================================================================
# 4. vLLM Embedding Server
# =============================================================================
echo "=> Uruchamianie vLLM Embedding Server na $EMBED_NODE:$EMBED_PORT ..."
srun --het-group=1 --overlap \
    $VENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL" \
    --host 0.0.0.0 \
    --port $EMBED_PORT \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.4 &
VLLM_EMBED_PID=$!

while ! curl -s http://$EMBED_NODE:$EMBED_PORT/v1/models > /dev/null; do sleep 5; done
echo "=> vLLM embedding server online."

# =============================================================================
# 5. vLLM Reranker Server
# =============================================================================
echo "=> Uruchamianie vLLM Reranker Server na $EMBED_NODE:$RERANK_PORT ..."
srun --het-group=1 --overlap \
    vllm serve "$RERANK_MODEL" \
    --host 0.0.0.0 \
    --port $RERANK_PORT \
    --gpu-memory-utilization 0.4 \
    --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' &
VLLM_RERANK_PID=$!

while ! curl -s http://$EMBED_NODE:$RERANK_PORT/v1/models > /dev/null; do sleep 5; done
echo "=> vLLM reranker server online."

# =============================================================================
# 6. Eval-only pass — CLEAN
# =============================================================================
echo ""
echo "=> Uruchamianie eval-only (CLEAN, noise=${NOISE_PROBABILITY}, n_samples=${N_SAMPLES})..."

srun --het-group=0 \
    $VENV_PYTHON -m marti.cli.multi_agent_train_ppo_ray \
    --pretrain "$TRAIN_MODEL" \
    --save_path "$OUTPUT_DIR" \
    --agents "{
        \"0\": {
            \"agent_id\": \"shared_agent\",
            \"role\": \"generator\",
            \"agent_role\": \"generator\",
            \"pretrain\": \"$TRAIN_MODEL\",
            \"is_tuning\": false,
            \"is_reasoning_model\": true
        }
    }" \
    --workflow_func_path "$WORKFLOW_SCRIPT" \
    --prompt_data "$EVAL_DATA_PATH" \
    --eval_dataset "$EVAL_DATA_PATH" \
    --eval_split train \
    --eval_steps 9999 \
    --eval_before_training \
    --eval_n_samples_per_prompt 1 \
    --workflow_args "{
        \"debug_dir\": \"$DEBUG_DIR\",
        \"embed_host\": \"$EMBED_NODE\",
        \"embed_port\": $EMBED_PORT,
        \"rerank_host\": \"$EMBED_NODE\",
        \"rerank_port\": $RERANK_PORT,
        \"similarity_weight\": $SIMILARITY_WEIGHT,
        \"diversity_weight\": $DIVERSITY_WEIGHT,
        \"groundedness_weight\": $GROUNDEDNESS_WEIGHT,
        \"relevancy_weight\": $RELEVANCY_WEIGHT,
        \"weaviate_url\": \"http://$WEAVIATE_NODE_ID:8080\",
        \"debug\": \"true\",
        \"use_weaviate_context\": \"$USE_WEAVIATE_CONTEXT\",
        \"weaviate_top_n\": $WEAVIATE_TOP_N,
        \"ask_retriever_limit\": $ASK_RETRIEVER_LIMIT,
        \"retriever_search_limit\": $RETRIEVER_SEARCH_LIMIT,
        \"apply_length_penalty\": \"false\",
        \"length_penalty_lambda\": 0.0,
        \"apply_channel_noise\": \"$APPLY_CHANNEL_NOISE\",
        \"noise_probability\": $NOISE_PROBABILITY
    }" \
    --input_key "user_query" \
    --label_key "hypothesis" \
    --metadata_key "metadata" \
    --advantage_estimator "group_norm" \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.4 \
    --colocate_all_models \
    --vllm_sync_backend nccl \
    --enforce_eager \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --lr_scheduler constant \
    --actor_learning_rate 0.0 \
    --train_batch_size 1 \
    --micro_train_batch_size 1 \
    --rollout_batch_size "$N_SAMPLES" \
    --n_samples_per_prompt "$N_SAMPLES" \
    --num_episodes 1 \
    --max_epochs 1 \
    --prompt_max_len 4000 \
    --generate_max_len 3000 \
    --eval_generate_max_len 3000 \
    --zero_stage 3 \
    --bf16 \
    --gradient_checkpointing \
    --seed 42

# =============================================================================
# 7. Cleanup
# =============================================================================
kill "$VLLM_EMBED_PID"  2>/dev/null || true
kill "$VLLM_RERANK_PID" 2>/dev/null || true

echo ""
echo "========================================================"
echo " CLEAN eval complete."
echo " Trajektorie: $DEBUG_DIR/eval/"
echo ""
echo " Gdy run_eval_noisy.sh również się skończy, uruchom:"
echo "   python src/eval/eval_noise_probe.py \\"
echo "     --noisy_dir $BASE_DIR/workflow_logs/$PROBE_NAME/eval_noisy \\"
echo "     --clean_dir $DEBUG_DIR \\"
echo "     --output_dir $BASE_DIR/eval_results/$PROBE_NAME"
echo "========================================================"