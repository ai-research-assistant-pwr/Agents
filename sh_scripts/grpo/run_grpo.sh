#!/bin/bash
# =================================================
# HET GROUP 0: Primary Training Node (2 GPUs)
# =================================================
#SBATCH --job-name=grpo_qwen
#SBATCH --output=/home/%u/disk/patryk/Agents/out/%x_%j.out
#SBATCH --time=0-00:10:00
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128gb
#SBATCH --gres=gpu:hopper:2
#SBATCH --ntasks-per-node=1

#SBATCH hetjob

# =================================================
# HET GROUP 1: Smaller Embedding Node (1 GPU)
# =================================================
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:hopper:1
#SBATCH --ntasks-per-node=1

set -e 

# 1. Command Line Arguments
WANDB_API_KEY=$1
WEAVIATE_NODE_ID=$2

# 2. Flexible Bash Variables (Defaults applied if not provided)
MY_DISK="${MY_DISK:-/home/$USER/disk}"
BASE_DIR="${BASE_DIR:-$MY_DISK/patryk/Agents}"
TRAIN_MODEL="${TRAIN_MODEL:-"Qwen/Qwen3-0.6B"}"
EMBED_MODEL="${EMBED_MODEL:-"Qwen/Qwen3-Embedding-4B"}"
WANDB_RUN="${WANDB_RUN_NAME:-$SLURM_JOB_NAME}" # Defaults to 'grpo_qwen'
BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ROLLOUT_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT:-1.0}"
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-1.0}"
DEBUG_WORKFLOW="${DEBUG_WORKFLOW:-true}"
USE_WEAVIATE_CONTEXT="${USE_WEAVIATE_CONTEXT:-true}"
WEAVIATE_TOP_N="${WEAVIATE_TOP_N:-5}"
ASK_RETRIEVER_LIMIT="${ASK_RETRIEVER_LIMIT:-1}"
RETRIEVER_SEARCH_LIMIT="${RETRIEVER_SEARCH_LIMIT:-1}"

# 3. Derived Paths
VENV_PATH="$BASE_DIR/venv"
AGENTS_DIR="$BASE_DIR"
MARTI_DIR="$BASE_DIR/MARTI"
DATA_PATH="$AGENTS_DIR/data/rl_grounded_dataset_merged.csv"
WORKFLOW_SCRIPT="$AGENTS_DIR/src/grpo/grpo_train/scientific_workflow.py"
OUTPUT_DIR="${OUTPUT_DIR:-$MY_DISK/patryk/models_output/${SLURM_JOB_NAME}_results}"

source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0

source $VENV_PATH/bin/activate
VENV_PYTHON="$VENV_PATH/bin/python"

export MY_NEW_TMP="$MY_DISK/patryk/tmp"
export XDG_CACHE_HOME="$MY_NEW_TMP/xdg_cache"
export TRITON_CACHE_DIR="$MY_NEW_TMP/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$MY_NEW_TMP/torchinductor_cache"
export VLLM_USE_V1="0"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

mkdir -p "$MY_NEW_TMP" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$OUTPUT_DIR"

# =================================================
# NODE DISCOVERY FOR HETJOBS
# =================================================
TRAIN_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0 | head -n 1)
EMBED_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1 | head -n 1)
EMBED_PORT=8000

echo "=> Job distributed across heterogeneous nodes:"
echo "   Trainer Node (2 GPUs): $TRAIN_NODE"
echo "   Embedding Node (1 GPU): $EMBED_NODE"
echo "   Train Model: $TRAIN_MODEL"
echo "   Embed Model: $EMBED_MODEL"

# =================================================
# START vLLM EMBEDDING SERVER (On Het Group 1)
# =================================================
echo "=> Starting vLLM Embedding Server on $EMBED_NODE..."

srun --het-group=1 \
    $VENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL" \
    --host 0.0.0.0 \
    --port $EMBED_PORT \
    --max-model-len 4096 &
VLLM_PID=$!

echo "=> Waiting for vLLM server to become ready..."
while ! curl -s http://$EMBED_NODE:$EMBED_PORT/v1/models > /dev/null; do
    sleep 5
done
echo "=> vLLM server is online at http://$EMBED_NODE:$EMBED_PORT/v1!"

# =================================================
# START nvidia-smi LOGGING (On Het Group 0)
# =================================================
NVIDIA_SMI_LOG="$BASE_DIR/logs/nvidia_smi_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$NVIDIA_SMI_LOG")"
(while true; do echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$NVIDIA_SMI_LOG"; nvidia-smi >> "$NVIDIA_SMI_LOG"; sleep 10; done) &
NVIDIA_SMI_PID=$!
echo "=> nvidia-smi logging started -> $NVIDIA_SMI_LOG"

# =================================================
# START TRAINING (On Het Group 0)
# =================================================
DEFAULT_AGENT="{
    \"is_reasoning_model\": true
}"

# Dynamically injected the training model variable
AGENT0="{
    \"0\": {
        \"agent_id\": \"shared_agent\",
        \"role\": \"generator\",
        \"agent_role\": \"generator\",
        \"pretrain\": \"$TRAIN_MODEL\",
        \"is_tuning\": true,
        \"is_reasoning_model\": true
    }
}"

echo "=> Running MARTI GRPO training (Workflow Mode)..."

# max_num_nodes = total trajectory records per prompt = (K+1)*(S+2)
# where K=ASK_RETRIEVER_LIMIT and S=RETRIEVER_SEARCH_LIMIT
MAX_NUM_NODES=$(( (ASK_RETRIEVER_LIMIT + 1) * (RETRIEVER_SEARCH_LIMIT + 2) ))
echo "   max_num_nodes (trajectory steps per prompt): $MAX_NUM_NODES"

srun --het-group=0 \
    $VENV_PYTHON -m marti.cli.multi_agent_train_ppo_ray \
    --pretrain "$TRAIN_MODEL" \
    --save_path "$OUTPUT_DIR" \
    --agents "$AGENT0" \
    --workflow_func_path "$WORKFLOW_SCRIPT" \
    --prompt_data "$DATA_PATH" \
    --workflow_args "{\"debug_dir\": \"$BASE_DIR/workflow_logs\", \"embed_host\": \"$EMBED_NODE\", \"embed_port\": $EMBED_PORT, \"similarity_weight\": $SIMILARITY_WEIGHT, \"diversity_weight\": $DIVERSITY_WEIGHT, \"weaviate_url\": \"http://$WEAVIATE_NODE_ID:8080\", \"debug\": \"$DEBUG_WORKFLOW\", \"use_weaviate_context\": \"$USE_WEAVIATE_CONTEXT\", \"weaviate_top_n\": $WEAVIATE_TOP_N, \"ask_retriever_limit\": $ASK_RETRIEVER_LIMIT, \"retriever_search_limit\": $RETRIEVER_SEARCH_LIMIT, \"max_num_nodes\": $MAX_NUM_NODES}" \
    --input_key "user_query" \
    --label_key "hypothesis" \
    --metadata_key "metadata" \
    --advantage_estimator "group_norm" \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.6 \
    --colocate_all_models \
    --vllm_sync_backend nccl \
    --enforce_eager \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --lr_scheduler constant \
    --actor_learning_rate 5e-7 \
    --use_kl_loss \
    --init_kl_coef 0.05 \
    --train_batch_size "$BATCH_SIZE" \
    --micro_train_batch_size 1 \
    --rollout_batch_size "$ROLLOUT_SIZE" \
    --n_samples_per_prompt 16 \
    --num_episodes 1 \
    --max_epochs 1 \
    --prompt_max_len 8096 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --gradient_checkpointing \
    --packing_samples \
    --save_hf_ckpt \
    --seed 42 \
    --logging_steps 1 \
    --use_wandb "$WANDB_API_KEY" \
    --wandb_project MARTI_GRPO \
    --wandb_run_name "$WANDB_RUN"

echo "=> Training completed successfully!"

# Stop background processes
kill "$NVIDIA_SMI_PID" 2>/dev/null
kill "$VLLM_PID" 2>/dev/null
echo "=> Background logging and vLLM processes stopped."