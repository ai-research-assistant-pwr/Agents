#!/bin/bash
# =============================================================================
# run_experiment.sh — Master Experiment Launcher
# =============================================================================
#
# Usage:
#   sbatch sh_scripts/grpo/run_experiment.sh <PRESET> [WANDB_API_KEY]
#
# Presets:
#   baseline_free      – No length penalty, no channel noise
#   exp_bottleneck     – Information Bottleneck only (length penalty)
#   exp_noise          – Channel noise only
#   exp_full           – Both IB + channel noise (primary experiment)
#
# Environment variables you can override before calling sbatch:
#   TRAIN_MODEL, EMBED_MODEL, TRAIN_BATCH_SIZE, ROLLOUT_BATCH_SIZE
#   SIMILARITY_WEIGHT, DIVERSITY_WEIGHT
#   LENGTH_PENALTY_LAMBDA, NOISE_PROBABILITY
#
# Examples:
#   sbatch sh_scripts/grpo/run_experiment.sh baseline_free  YOUR_WANDB_KEY
#   sbatch sh_scripts/grpo/run_experiment.sh exp_full       YOUR_WANDB_KEY
#
#   # Override batch size for a quick smoke test:
#   TRAIN_BATCH_SIZE=2 ROLLOUT_BATCH_SIZE=2 \
#     sbatch sh_scripts/grpo/run_experiment.sh exp_full YOUR_WANDB_KEY
#
# =============================================================================
#SBATCH --job-name=grpo_experiment
#SBATCH --output=Agents/out/%x_%j.out
#SBATCH --time=0-04:00:00
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128gb
#SBATCH --gres=gpu:hopper:2
#SBATCH --ntasks-per-node=1

#SBATCH hetjob

#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:hopper:1
#SBATCH --ntasks-per-node=1

set -e

# =============================================================================
# 1. Arguments
# =============================================================================
PRESET="${1:-exp_full}"
WANDB_API_KEY="${2:-}"

if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: No WANDB_API_KEY supplied. WandB logging will be disabled."
fi

echo "========================================================"
echo " Experiment preset : $PRESET"
echo " SLURM Job ID      : $SLURM_JOB_ID"
echo "========================================================"

# =============================================================================
# 2. Preset Configuration
# =============================================================================
# Defaults (overridden per preset below)
APPLY_LENGTH_PENALTY="false"
LENGTH_PENALTY_LAMBDA="${LENGTH_PENALTY_LAMBDA:-0.001}"
APPLY_CHANNEL_NOISE="false"
NOISE_PROBABILITY="${NOISE_PROBABILITY:-0.15}"
SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT:-1.0}"
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-1.0}"

case "$PRESET" in

  # --------------------------------------------------------------------------
  # baseline_free: no constraints, direct communication
  # --------------------------------------------------------------------------
  baseline_free)
    APPLY_LENGTH_PENALTY="false"
    APPLY_CHANNEL_NOISE="false"
    RUN_SUFFIX="baseline_free"
    ;;

  # --------------------------------------------------------------------------
  # exp_bottleneck: Information Bottleneck — length penalty only
  # --------------------------------------------------------------------------
  exp_bottleneck)
    APPLY_LENGTH_PENALTY="true"
    APPLY_CHANNEL_NOISE="false"
    RUN_SUFFIX="exp_bottleneck"
    ;;

  # --------------------------------------------------------------------------
  # exp_noise: Channel noise only
  # --------------------------------------------------------------------------
  exp_noise)
    APPLY_LENGTH_PENALTY="false"
    APPLY_CHANNEL_NOISE="true"
    RUN_SUFFIX="exp_noise"
    ;;

  # --------------------------------------------------------------------------
  # exp_full: Both IB + channel noise (primary experiment)
  # --------------------------------------------------------------------------
  exp_full)
    APPLY_LENGTH_PENALTY="true"
    APPLY_CHANNEL_NOISE="true"
    RUN_SUFFIX="exp_full"
    ;;

  *)
    echo "ERROR: Unknown preset '$PRESET'."
    echo "Valid presets: baseline_free | exp_bottleneck | exp_noise | exp_full"
    exit 1
    ;;
esac

# =============================================================================
# 3. Paths & Environment
# =============================================================================
MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"

TRAIN_MODEL="${TRAIN_MODEL:-"Qwen/Qwen3-0.6B"}"
EMBED_MODEL="${EMBED_MODEL:-"Qwen/Qwen3-Embedding-4B"}"
BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ROLLOUT_SIZE="${ROLLOUT_BATCH_SIZE:-8}"

# Separate output and log dirs per run — critical for eval scripts
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${RUN_SUFFIX}_${TIMESTAMP}"
OUTPUT_DIR="$BASE_DIR/models_output/${RUN_NAME}"
WORKFLOW_LOGS_DIR="$BASE_DIR/workflow_logs/${RUN_NAME}"

VENV_PATH="$BASE_DIR/venv"
WORKFLOW_SCRIPT="$BASE_DIR/src/grpo/grpo_train/scientific_workflow.py"

DATASET_PATH_TYM="$BASE_DIR/data/datasets/rl_grounded_dataset_merged.csv"
DATASET_PATH_PAT="/home/patswi3426/disk/patryk/Agents/data/rl_grounded_dataset_merged.csv"

if [ -r "$DATASET_PATH_TYM" ]; then
    DATA_PATH="$DATASET_PATH_TYM"
elif [ -r "$DATASET_PATH_PAT" ]; then
    DATA_PATH="$DATASET_PATH_PAT"
else
    echo "ERROR: Dataset not found."
    exit 1
fi

source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0
source $VENV_PATH/bin/activate
VENV_PYTHON="$VENV_PATH/bin/python"

MY_NEW_TMP="$MY_DISK/tmp"
export XDG_CACHE_HOME="$MY_NEW_TMP/xdg_cache"
export TRITON_CACHE_DIR="$MY_NEW_TMP/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$MY_NEW_TMP/torchinductor_cache"
export VLLM_USE_V1="0"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

mkdir -p "$MY_NEW_TMP" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$OUTPUT_DIR" "$WORKFLOW_LOGS_DIR" \
         "$BASE_DIR/logs"

# =============================================================================
# 4. Node Discovery
# =============================================================================
TRAIN_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0 | head -n 1)
EMBED_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1 | head -n 1)
EMBED_PORT=8000

echo ""
echo "  Preset              : $PRESET"
echo "  Run name            : $RUN_NAME"
echo "  Trainer node        : $TRAIN_NODE (2 GPUs)"
echo "  Embedding node      : $EMBED_NODE (1 GPU)"
echo "  Train model         : $TRAIN_MODEL"
echo "  Embed model         : $EMBED_MODEL"
echo "  Output dir          : $OUTPUT_DIR"
echo "  Workflow logs dir   : $WORKFLOW_LOGS_DIR"
echo "  Apply length penalty: $APPLY_LENGTH_PENALTY (λ=$LENGTH_PENALTY_LAMBDA)"
echo "  Apply channel noise : $APPLY_CHANNEL_NOISE (p=$NOISE_PROBABILITY)"
echo ""

# Save run config for reproducibility
cat > "$OUTPUT_DIR/run_config.json" <<EOF
{
  "preset":                "$PRESET",
  "run_name":              "$RUN_NAME",
  "slurm_job_id":          "$SLURM_JOB_ID",
  "timestamp":             "$TIMESTAMP",
  "train_model":           "$TRAIN_MODEL",
  "embed_model":           "$EMBED_MODEL",
  "batch_size":            $BATCH_SIZE,
  "rollout_size":          $ROLLOUT_SIZE,
  "apply_length_penalty":  $APPLY_LENGTH_PENALTY,
  "length_penalty_lambda": $LENGTH_PENALTY_LAMBDA,
  "apply_channel_noise":   $APPLY_CHANNEL_NOISE,
  "noise_probability":     $NOISE_PROBABILITY,
  "similarity_weight":     $SIMILARITY_WEIGHT,
  "diversity_weight":      $DIVERSITY_WEIGHT,
  "data_path":             "$DATA_PATH"
}
EOF

# =============================================================================
# 5. Start vLLM Embedding Server
# =============================================================================
echo "=> Starting vLLM Embedding Server on $EMBED_NODE..."

srun --het-group=1 \
    $VENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL" \
    --host 0.0.0.0 \
    --port $EMBED_PORT \
    --max-model-len 4096 &
VLLM_PID=$!

# Wait with timeout
MAX_WAIT=120
WAITED=0
echo "=> Waiting for vLLM server (timeout ${MAX_WAIT}s)..."
while ! curl -s http://$EMBED_NODE:$EMBED_PORT/v1/models > /dev/null 2>&1; do
    sleep 5; WAITED=$((WAITED+5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s. Aborting."
        kill "$VLLM_PID" 2>/dev/null
        exit 1
    fi
done
echo "=> vLLM server online at http://$EMBED_NODE:$EMBED_PORT/v1"

# =============================================================================
# 6. nvidia-smi Logging
# =============================================================================
NVIDIA_SMI_LOG="$BASE_DIR/logs/nvidia_smi_${SLURM_JOB_ID}_${RUN_SUFFIX}.log"
(while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$NVIDIA_SMI_LOG"
    nvidia-smi >> "$NVIDIA_SMI_LOG"
    sleep 10
done) &
NVIDIA_SMI_PID=$!

# =============================================================================
# 7. Training
# =============================================================================
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

echo "=> Running MARTI GRPO training — preset: $PRESET"

srun --het-group=0 \
    $VENV_PYTHON -m marti.cli.multi_agent_train_ppo_ray \
    --pretrain "$TRAIN_MODEL" \
    --save_path "$OUTPUT_DIR" \
    --agents "$AGENT0" \
    --workflow_func_path "$WORKFLOW_SCRIPT" \
    --prompt_data "$DATA_PATH" \
    --workflow_args "{
        \"debug_dir\": \"$WORKFLOW_LOGS_DIR\",
        \"embed_host\": \"$EMBED_NODE\",
        \"embed_port\": $EMBED_PORT,
        \"similarity_weight\": $SIMILARITY_WEIGHT,
        \"diversity_weight\": $DIVERSITY_WEIGHT,
        \"apply_length_penalty\": $APPLY_LENGTH_PENALTY,
        \"length_penalty_lambda\": $LENGTH_PENALTY_LAMBDA,
        \"apply_channel_noise\": $APPLY_CHANNEL_NOISE,
        \"noise_probability\": $NOISE_PROBABILITY
    }" \
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
    --wandb_run_name "$RUN_NAME"

echo "=> Training completed: $RUN_NAME"

# =============================================================================
# 8. Write eval manifest (used by eval scripts to find logs)
# =============================================================================
cat > "$BASE_DIR/eval_results/.last_run_${PRESET}.json" <<EOF
{
  "preset":          "$PRESET",
  "run_name":        "$RUN_NAME",
  "workflow_logs":   "$WORKFLOW_LOGS_DIR",
  "model_output":    "$OUTPUT_DIR",
  "embed_node":      "$EMBED_NODE",
  "embed_port":      $EMBED_PORT
}
EOF

# =============================================================================
# 9. Cleanup
# =============================================================================
kill "$NVIDIA_SMI_PID" 2>/dev/null
kill "$VLLM_PID" 2>/dev/null
echo "=> Background processes stopped."
echo "=> Run manifest: $BASE_DIR/eval_results/.last_run_${PRESET}.json"