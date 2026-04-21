#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=0-04:00:00
#SBATCH --job-name=grpo_qwen
#SBATCH --output=/home/tymrom7227/disk/Agents/out/grpo_qwen.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:2

set -e 

WANDB_API_KEY=$1

# =================================================
# ENV SETUP
# =================================================
source /usr/local/sbin/modules.sh
module load Python/3.11.5-GCCcore-13.2.0

MY_DISK="/home/tymrom7227/disk"
VENV_PATH="$MY_DISK/venvs/pnw-3"
AGENTS_DIR="$MY_DISK/Agents"
MARTI_DIR="$MY_DISK/MARTI"
MERGED_MODEL="$MY_DISK/models_output/run4/Qwen3-4B-SFT-Merged"

source $VENV_PATH/bin/activate
VENV_PYTHON="$VENV_PATH/bin/python"

export PYTHONPATH="$MARTI_DIR:$AGENTS_DIR:$PYTHONPATH"
export XDG_CACHE_HOME=$MY_DISK/.cache

# =================================================
# VALIDATION
# =================================================
if [ ! -f "$MERGED_MODEL/config.json" ]; then
    echo "ERROR: Model not found at $MERGED_MODEL"
    exit 1
fi

if [ ! -f "$AGENTS_DIR/config/grpo/config.yaml" ]; then
    echo "ERROR: Config not found at $AGENTS_DIR/config/grpo/config.yaml"
    exit 1
fi

if [ ! -f "$AGENTS_DIR/src/grpo/grpo_train/environment.py" ]; then
    echo "ERROR: Environment script not found at $AGENTS_DIR/src/grpo/grpo_train/environment.py"
    exit 1
fi

# =================================================
# START TRAINING
# =================================================
DATA_PATH="$AGENTS_DIR/data/datasets/grpo_exp_dataset/mock_data.json"
AGENT_ENV_SCRIPT="$AGENTS_DIR/src/grpo/grpo_train/environment.py"
OUTPUT_DIR="$MY_DISK/models_output/grpo_results"
export MARTI_CONFIG_PATH="$AGENTS_DIR/config/grpo/config.yaml"

mkdir -p "$OUTPUT_DIR"

if [ ! -z "$WANDB_API_KEY" ]; then
    WANDB_FLAG="--use_wandb $WANDB_API_KEY --wandb_project MARTI_GRPO --wandb_run_name grpo_exp_1"
fi

echo "=> Configuration:"
echo "   Model: $MERGED_MODEL"
echo "   Data: $DATA_PATH"
echo "   Agent: $AGENT_ENV_SCRIPT"
echo "   Output: $OUTPUT_DIR"
echo "   Config: $MARTI_CONFIG_PATH"
echo ""
echo "=> Running MARTI GRPO training..."
$VENV_PYTHON -m marti.cli.train_ppo_ray \
    --pretrain "$MERGED_MODEL" \
    --save_path "$OUTPUT_DIR" \
    --agent_func_path "$AGENT_ENV_SCRIPT" \
    --prompt_data "$DATA_PATH" \
    --input_key "query" \
    --label_key "expected_action" \
    --advantage_estimator "group_norm" \
    --colocate_actor_ref \
    --colocate_critic_reward \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_enable_sleep \
    --enforce_eager \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 5e-6 \
    --train_batch_size 16 \
    --micro_train_batch_size 1 \
    --rollout_batch_size 16 \
    --n_samples_per_prompt 4 \
    --max_epochs 10 \
    --prompt_max_len 1024 \
    --generate_max_len 256 \
    --max_len 2048 \
    --zero_stage 3 \
    --bf16 \
    --gradient_checkpointing \
    --save_hf_ckpt \
    --seed 42 \
    --logging_steps 1 \
    $WANDB_FLAG

echo "=> Training completed successfully!"