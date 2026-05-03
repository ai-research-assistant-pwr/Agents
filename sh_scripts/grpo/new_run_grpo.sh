#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128gb
#SBATCH --time=0-00:05:00
#SBATCH --job-name=grpo_qwen
#SBATCH --output=/home/patswi3426/disk/patryk/Agents/out/grpo_qwen.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:3

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

# =================================================
# START TRAINING
# =================================================
DATA_PATH="$AGENTS_DIR/data/rl_grounded_dataset_v2.csv"
WORKFLOW_SCRIPT="$AGENTS_DIR/src/grpo/grpo_train/scientific_workflow.py"
OUTPUT_DIR="$MY_DISK/patryk/models_output/grpo_results"

mkdir -p "$OUTPUT_DIR"

# =================================================
# START nvidia-smi LOGGING
# =================================================
NVIDIA_SMI_LOG="$MY_DISK/patryk/Agents/logs/nvidia_smi_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$NVIDIA_SMI_LOG")"
watch -n 10 "echo \"=== \$(date '+%Y-%m-%d %H:%M:%S') ===\" >> \"$NVIDIA_SMI_LOG\"; nvidia-smi >> \"$NVIDIA_SMI_LOG\"" &
NVIDIA_SMI_PID=$!
echo "=> nvidia-smi logging started (PID=$NVIDIA_SMI_PID) -> $NVIDIA_SMI_LOG"

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

echo "=> Configuration:"
echo "   Data: $DATA_PATH"
echo "   Workflow: $WORKFLOW_SCRIPT"
echo "   Output: $OUTPUT_DIR"
echo ""
echo "=> Running MARTI GRPO training (Workflow Mode)..."

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
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.6 \
    --colocate_all_models \
    --vllm_sync_backend nccl \
    --enforce_eager \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --lr_scheduler constant \
    --actor_learning_rate 5e-7 \
    --use_kl_loss \
    --init_kl_coef 0.05 \
    --train_batch_size 4 \
    --micro_train_batch_size 1 \
    --rollout_batch_size 4 \
    --n_samples_per_prompt 8 \
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
    --wandb_run_name grpo_multiagent_1


echo "=> Training completed successfully!"

# Stop nvidia-smi logging
kill "$NVIDIA_SMI_PID" 2>/dev/null
echo "=> nvidia-smi logging stopped."