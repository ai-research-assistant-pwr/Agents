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
# AUTOMATYCZNE MERGOWANIE MODELU (LORA + BASE)
# =================================================
BASE_MODEL="$MY_DISK/models/Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER="$MY_DISK/models_output/run4/lora_generator_Qwen3-4B-Instruct-2507"
MERGED_MODEL="$MY_DISK/models_output/run4/Qwen3-4B-SFT-Merged"

if [ ! -d "$MERGED_MODEL" ]; then
    echo "=> No merged model found. Starting LoRA weight merging on CPU..."
    $VENV_PYTHON << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("1/4 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained("$BASE_MODEL", torch_dtype=torch.bfloat16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained("$BASE_MODEL")

print("2/4 Applying SFT (LoRA) weights...")
model = PeftModel.from_pretrained(base_model, "$SFT_ADAPTER")

print("3/4 Merging weights into base model...")
model = model.merge_and_unload()

print("4/4 Saving to disk...")
model.save_pretrained("$MERGED_MODEL")
tokenizer.save_pretrained("$MERGED_MODEL")
print("Success! Weights merged.")
EOF
else
    echo "=> Merged model found: $MERGED_MODEL. Skipping merging."
fi

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

echo "=> Running GRPO training..."
$VENV_PYTHON -m openrlhf.cli.train_ppo_ray \
    --actor.model_name_or_path $MERGED_MODEL \
    --ckpt.output_dir $OUTPUT_DIR \
    --train.agent_func_path $AGENT_ENV_SCRIPT \
    --data.prompt_dataset "$DATA_PATH" \
    --data.input_key "query" \
    --data.label_key "expected_action" \
    --algo.advantage.estimator "group_norm" \
    --colocate_actor_ref \
    --vllm.num_engines 1 \
    --vllm.tensor_parallel_size 1 \
    --vllm.gpu_memory_utilization 0.4 \
    --vllm.enable_sleep \
    --vllm.enforce_eager \
    --actor.num_nodes 1 \
    --actor.num_gpus_per_node 0.4 \
    --ref.num_nodes 1 \
    --ref.num_gpus_per_node 0.4 \
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