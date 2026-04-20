#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=0-04:00:00
#SBATCH --job-name=grpo_qwen
#SBATCH --output=Agents/out/grpo_qwen.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

set -e  #

WANDB_API_KEY=$1

if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: No WANDB API key provided. Running without wandb."
    WANDB_FLAG=""
else
    echo "WANDB API key provided. Logging to Weights & Biases."
    WANDB_FLAG="--use_wandb $WANDB_API_KEY --wandb_project MARTI_GRPO --wandb_run_name grpo_exp_1"
fi

# =================================================
# ENV SETUP
# =================================================
source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="/home/tymrom7227/disk/venvs/pnw-3"
source $VENV_PATH/bin/activate
VENV_PYTHON="$VENV_PATH/bin/python"

echo "Using Python: $VENV_PYTHON"

# =================================================
# AUTO-INSTALL OPENRLHF (if missing)
# =================================================
echo "Checking openrlhf installation..."

if ! python -c "import openrlhf" &> /dev/null; then
    echo "openrlhf not found. Installing..."

    pip install openrlhf || true

    LOCAL_OPENRLHF="/home/tymrom7227/disk/Agents/openrlhf"

    if [ -d "$LOCAL_OPENRLHF" ]; then
        echo "Installing openrlhf from local repo..."
        pip install -e $LOCAL_OPENRLHF
    fi
fi

# =================================================
# FINAL CHECK
# =================================================
echo "Verifying openrlhf import..."

python - <<EOF
import openrlhf
print("openrlhf successfully imported")
EOF

# =================================================
# PATH CONFIG
# =================================================
MY_DISK="/home/tymrom7227/disk"
AGENTS_DIR="$MY_DISK/Agents"

export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

export XDG_CACHE_HOME=$MY_DISK/.cache
export HF_HOME=$MY_DISK/.cache/hf
export TORCHINDUCTOR_CACHE_DIR=$MY_DISK/.cache/torch_inductor
export TRANSFORMERS_OFFLINE=0

# =================================================
# PATHS
# =================================================
BASE_MODEL="$MY_DISK/models/Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER="$MY_DISK/models_output/run4/lora_generator_Qwen3-4B-Instruct-2507"
DATA_PATH="$AGENTS_DIR/data/datasets/grpo_exp_dataset/mock_data.json"
AGENT_ENV_SCRIPT="$AGENTS_DIR/src/grpo/grpo_train/environment.py"
OUTPUT_DIR="$MY_DISK/models_output/grpo_results"

export MARTI_CONFIG_PATH="$AGENTS_DIR/config.yaml"

# =================================================
# SANITY CHECK
# =================================================
echo "================================================="
echo "TESTING CONFIGURATION BEFORE GRPO TRAINING"
echo "================================================="

FILES_TO_CHECK=(
    "$BASE_MODEL"
    "$SFT_ADAPTER/adapter_config.json"
    "$DATA_PATH"
    "$AGENT_ENV_SCRIPT"
    "$MARTI_CONFIG_PATH"
)

for FILE in "${FILES_TO_CHECK[@]}"; do
    if [ ! -e "$FILE" ]; then
        echo "CRITICAL ERROR: Path not found: $FILE"
        exit 1
    else
        echo "Path verified: $FILE"
    fi
done

# =================================================
# START TRAINING
# =================================================
echo "================================================="
echo "Starting GRPO training via MARTI / OpenRLHF..."
echo "================================================="

$VENV_PYTHON -m openrlhf.cli.train_ppo_ray \
    --pretrain $BASE_MODEL \
    --adapter_path $SFT_ADAPTER \
    --save_path $OUTPUT_DIR \
    --agent_func_path $AGENT_ENV_SCRIPT \
    --prompt_data "json@$DATA_PATH" \
    --input_key "query" \
    --label_key "expected_action" \
    --observation_key "visible_chunks" \
    --advantage_estimator "group_norm" \
    --colocate_all_models \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 5e-6 \
    --train_batch_size 16 \
    --micro_train_batch_size 1 \
    --rollout_batch_size 16 \
    --n_samples_per_prompt 4 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 256 \
    --zero_stage 3 \
    --bf16 \
    --gradient_checkpointing \
    --save_hf_ckpt \
    $WANDB_FLAG

echo "====================================="
echo "GRPO Training completed successfully!"
echo "====================================="