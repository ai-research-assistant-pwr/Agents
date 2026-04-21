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

echo "=> Odinstalowanie bazowego OpenRLHF aby wymusic uzycie biblioteki MARTI..."
$VENV_PYTHON -m pip uninstall -y openrlhf

export PYTHONPATH="$MARTI_DIR:$AGENTS_DIR:$PYTHONPATH"
export XDG_CACHE_HOME=$MY_DISK/.cache

# =================================================
# START TRAINING
# =================================================
DATA_PATH="$AGENTS_DIR/data/datasets/grpo_exp_dataset/mock_data.json"
AGENT_ENV_SCRIPT="$AGENTS_DIR/src/grpo/grpo_train/environment.py"
OUTPUT_DIR="$MY_DISK/models_output/grpo_results"
export MARTI_CONFIG_PATH="$AGENTS_DIR/config/grpo/config.yaml"

if [ ! -z "$WANDB_API_KEY" ]; then
    WANDB_FLAG="--logger.wandb.key $WANDB_API_KEY --logger.wandb.project MARTI_GRPO --logger.wandb.run_name grpo_exp_1"
fi

echo "=> Running MARTI GRPO training..."
$VENV_PYTHON -m marti.cli.train_ppo_ray \
    --actor.model_name_or_path $MERGED_MODEL \
    --ckpt.output_dir $OUTPUT_DIR \
    --train.agent_func_path $AGENT_ENV_SCRIPT \
    --data.prompt_dataset "$DATA_PATH" \
    --data.input_key "query" \
    --data.label_key "expected_action" \
    --algo.advantage.estimator "group_norm" \
    --train.colocate_actor_ref \
    --vllm.num_engines 1 \
    --vllm.tensor_parallel_size 1 \
    --vllm.gpu_memory_utilization 0.2 \
    --vllm.enable_sleep \
    --vllm.enforce_eager \
    --actor.num_nodes 1 \
    --actor.num_gpus_per_node 1 \
    --ref.num_nodes 1 \
    --ref.num_gpus_per_node 1 \
    --actor.adam.lr 5e-7 \
    --train.batch_size 16 \
    --train.micro_batch_size 1 \
    --rollout.batch_size 16 \
    --rollout.n_samples_per_prompt 4 \
    --train.max_epochs 1 \
    --data.max_len 2048 \
    --rollout.max_new_tokens 256 \
    --ds.zero_stage 3 \
    --ds.param_dtype bf16 \
    --actor.gradient_checkpointing_enable \
    --ckpt.save_hf \
    $WANDB_FLAG