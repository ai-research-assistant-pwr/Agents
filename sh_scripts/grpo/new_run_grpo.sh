#!/bin/bash
# =================================================
# PACK GROUP 0: Primary Training Node (2 GPUs)
# =================================================
#SBATCH --job-name=grpo_qwen
#SBATCH --output=/home/patswi3426/disk/patryk/Agents/out/grpo_qwen.out
#SBATCH --time=0-00:05:00
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128gb
#SBATCH --gres=gpu:hopper:2
#SBATCH --ntasks-per-node=1

#SBATCH hetjob

# =================================================
# PACK GROUP 1: Smaller Embedding Node (1 GPU)
# =================================================
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:hopper:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/patswi3426/disk/patryk/Agents/out/grpo_embed.out

set -e 

WANDB_API_KEY=$1

source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0

MY_DISK="/home/patswi3426/disk/"
VENV_PATH="$MY_DISK/patryk/Agents/venv"
AGENTS_DIR="$MY_DISK/patryk/Agents/"
MARTI_DIR="$MY_DISK/patryk/Agents/MARTI"

source $VENV_PATH/bin/activate
VENV_PYTHON="$VENV_PATH/bin/python"

export MY_NEW_TMP="/mnt/lscratch/slurm/$SLURM_JOB_ID"
export XDG_CACHE_HOME="$MY_NEW_TMP/xdg_cache"
export TRITON_CACHE_DIR="$MY_NEW_TMP/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$MY_NEW_TMP/torchinductor_cache"
export VLLM_USE_V1="0"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

mkdir -p "$MY_NEW_TMP"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# =================================================
# NODE DISCOVERY FOR PACKJOBS
# =================================================
# SLURM automatically creates environment variables for each pack group
TRAIN_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0 | head -n 1)
EMBED_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1 | head -n 1)
EMBED_PORT=8000

echo "=> Job distributed across heterogeneous nodes:"
echo "   Trainer Node (2 GPUs): $TRAIN_NODE"
echo "   Embedding Node (1 GPU): $EMBED_NODE"

# =================================================
# START vLLM EMBEDDING SERVER (On Pack 1)
# =================================================
echo "=> Starting vLLM Embedding Server on $EMBED_NODE..."

# Use --het-group=1 to specifically target the 1-GPU node
srun --het-group=1 \
    $VENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-4B \
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
# START nvidia-smi LOGGING (On Pack 0)
# =================================================
NVIDIA_SMI_LOG="$MY_DISK/patryk/Agents/logs/nvidia_smi_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$NVIDIA_SMI_LOG")"
(while true; do echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$NVIDIA_SMI_LOG"; nvidia-smi >> "$NVIDIA_SMI_LOG"; sleep 10; done) &
NVIDIA_SMI_PID=$!
echo "=> nvidia-smi logging started -> $NVIDIA_SMI_LOG"

# =================================================
# START TRAINING (On Pack 0)
# =================================================
DATA_PATH="$AGENTS_DIR/data/rl_grounded_dataset_v2.csv"
WORKFLOW_SCRIPT="$AGENTS_DIR/src/grpo/grpo_train/scientific_workflow.py"
OUTPUT_DIR="$MY_DISK/patryk/models_output/grpo_results"

mkdir -p "$OUTPUT_DIR"

DEFAULT_AGENT="{
    \"is_reasoning_model\": true
}"

AGENT0="{
    \"0\": {
        \"agent_id\": \"shared_agent\",
        \"role\": \"generator\",
        \"agent_role\": \"generator\",
        \"pretrain\": \"Qwen/Qwen3-0.6B\",
        \"is_tuning\": true,
        \"is_reasoning_model\": true
    }
}"

echo "=> Running MARTI GRPO training (Workflow Mode)..."

# Use --het-group=0 to specifically target the 2-GPU node
srun --het-group=0 \
    $VENV_PYTHON -m marti.cli.multi_agent_train_ppo_ray \
    --pretrain "Qwen/Qwen3-0.6B" \
    --save_path "$OUTPUT_DIR" \
    --agents "$AGENT0" \
    --workflow_func_path "$WORKFLOW_SCRIPT" \
    --prompt_data "$DATA_PATH" \
    --workflow_args "{\"debug_dir\": \"$MY_DISK/patryk/Agents/logs\"}" \
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
    --train_batch_size 8 \
    --micro_train_batch_size 1 \
    --rollout_batch_size 8 \
    --n_samples_per_prompt 16 \
    --num_episodes 5 \
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
    --use_wandb $WANDB_API_KEY \
    --wandb_project MARTI_GRPO \
    --wandb_run_name grpo_multiagent_1 \
    --embedding_server_host $EMBED_NODE \
    --embedding_server_port $EMBED_PORT

echo "=> Training completed successfully!"

# Stop background processes
kill "$NVIDIA_SMI_PID" 2>/dev/null
kill "$VLLM_PID" 2>/dev/null
echo "=> Background logging and vLLM processes stopped."