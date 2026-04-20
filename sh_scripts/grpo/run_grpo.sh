#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=0-04:00:00
#SBATCH --job-name=grpo_qwen
#SBATCH --output=/home/tymrom7227/disk/Agents/out/grpo_qwen.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

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

source $VENV_PATH/bin/activate
VENV_PYTHON="$VENV_PATH/bin/python"

export PYTHONPATH="$MARTI_DIR:$AGENTS_DIR:$PYTHONPATH"

export XDG_CACHE_HOME=$MY_DISK/.cache
export HF_HOME=$MY_DISK/.cache/hf
export TRANSFORMERS_OFFLINE=0
export NCCL_DEBUG=WARN
export TRITON_CACHE_DIR="$MY_DISK/.cache/triton" 

# =================================================
# FINAL CHECKS
# =================================================
if ! $VENV_PYTHON -c "import openrlhf" &> /dev/null; then
    echo "MARTI not recognized, injecting .pth file directly into venv..."
    echo "$MARTI_DIR" > "$VENV_PATH/lib/python3.11/site-packages/marti.pth"
fi

# =================================================
# START TRAINING
# =================================================
BASE_MODEL="$MY_DISK/models/Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER="$MY_DISK/models_output/run4/lora_generator_Qwen3-4B-Instruct-2507"
DATA_PATH="$AGENTS_DIR/data/datasets/grpo_exp_dataset/mock_data.json"
AGENT_ENV_SCRIPT="$AGENTS_DIR/src/grpo/grpo_train/environment.py"
OUTPUT_DIR="$MY_DISK/models_output/grpo_results"
export MARTI_CONFIG_PATH="$AGENTS_DIR/config/grpo/config.yaml"

if [ ! -z "$WANDB_API_KEY" ]; then
    WANDB_FLAG="--logger.wandb.key $WANDB_API_KEY --logger.wandb.project MARTI_GRPO --logger.wandb.run_name grpo_exp_1"
fi

$VENV_PYTHON -m openrlhf.cli.train_ppo_ray \
    --actor.model_name_or_path $SFT_ADAPTER \
    --ckpt.output_dir $OUTPUT_DIR \
    --train.agent_func_path $AGENT_ENV_SCRIPT \
    --data.prompt_dataset "json@$DATA_PATH" \
    --data.input_key "query" \
    --data.label_key "expected_action" \
    --algo.advantage.estimator "group_norm" \
    --train.colocate_all \
    --vllm.num_engines 1 \
    --vllm.tensor_parallel_size 1 \
    --vllm.gpu_memory_utilization 0.5 \
    --vllm.enable_sleep \
    --ds.enable_sleep \
    --vllm.enforce_eager \
    --ref.num_nodes 1 \
    --ref.num_gpus_per_node 1 \
    --actor.num_nodes 1 \
    --actor.num_gpus_per_node 1 \
    --actor.adam.lr 5e-7 \
    --train.batch_size 16 \
    --train.micro_batch_size 1 \
    --rollout.batch_size 16 \
    --rollout.n_samples_per_prompt 4 \
    --train.max_epochs 1 \
    --data.max_len 2048 \
    --rollout.max_new_tokens 256 \
    --ds.zero_stage 2 \
    --ds.param_dtype bf16 \
    --actor.gradient_checkpointing_enable \
    --ckpt.save_hf \
    $WANDB_FLAG