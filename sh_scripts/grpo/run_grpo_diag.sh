#!/bin/bash
# =================================================
# DIAGNOSTIC RUN — weryfikacja poprawek EC
# ~50 kroków, 2 GPU, ~30-45 minut
#
# Co weryfikuje:
#   1. n_channel_tokens logowany w debug JSON        → fix turns.py
#   2. mean_channel_tokens widoczne w ec_stats       → fix workflow.py
#   3. eval reward NIE jest zaniżony przez szum      → fix is_eval
#   4. kara za długość jest widoczna w nagrodzie     → λ=0.005
#   5. pipeline w ogóle działa z K=1, S=1
# =================================================
#SBATCH --job-name=grpo_ec_diag
#SBATCH --output=Agents/out/%x_%j.out
#SBATCH --time=0-01:00:00
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
#SBATCH --ntasks-per-node=2

set -e

WANDB_API_KEY=$1
WEAVIATE_NODE_ID=$2

MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"

TRAIN_MODEL="${TRAIN_MODEL:-"Qwen/Qwen3-0.6B"}"
EMBED_MODEL="${EMBED_MODEL:-"Qwen/Qwen3-Embedding-4B"}"
RERANK_MODEL="${RERANK_MODEL:-"Qwen/Qwen3-Reranker-0.6B"}"
WANDB_RUN="${WANDB_RUN_NAME:-$SLURM_JOB_NAME}"

VENV_PATH="$BASE_DIR/venv"
AGENTS_DIR="$BASE_DIR"
TRAIN_DATA_PATH="$AGENTS_DIR/data/datasets/rl_grounded_dataset_train.csv"
EVAL_DATA_PATH="$AGENTS_DIR/data/datasets/rl_grounded_dataset_test.csv"
WORKFLOW_SCRIPT="$AGENTS_DIR/src/grpo/grpo_train/scientific_workflow.py"
OUTPUT_DIR="$BASE_DIR/models_output/${SLURM_JOB_NAME}_results"

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
mkdir -p "$MY_NEW_TMP" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$OUTPUT_DIR"

TRAIN_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0 | head -n 1)
EMBED_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1 | head -n 1)
EMBED_PORT=8000
RERANK_PORT=8001

echo "=== DIAGNOSTIC RUN ==="
echo "Train node : $TRAIN_NODE"
echo "Embed node : $EMBED_NODE"
echo "Checking: K=1 S=1 | lambda=0.005 | noise=0.15 | batch=4 | n_samples=4"

# --- Embedding server ---
srun --het-group=1 --overlap \
    $VENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL" \
    --host 0.0.0.0 --port $EMBED_PORT \
    --max-model-len 4096 --gpu-memory-utilization 0.4 &
VLLM_EMBED_PID=$!

echo "Waiting for embedding server..."
while ! curl -s http://$EMBED_NODE:$EMBED_PORT/v1/models > /dev/null; do sleep 5; done
echo "Embedding server ready."

# --- Reranker server ---
srun --het-group=1 --overlap \
    vllm serve "$RERANK_MODEL" \
    --host 0.0.0.0 --port $RERANK_PORT \
    --gpu-memory-utilization 0.4 \
    --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' &
VLLM_RERANK_PID=$!

echo "Waiting for reranker server..."
while ! curl -s http://$EMBED_NODE:$RERANK_PORT/v1/models > /dev/null; do sleep 5; done
echo "Reranker server ready."

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

# =================================================
# KLUCZOWE USTAWIENIA DIAGNOSTYCZNE:
#
#   train_batch_size=4, n_samples_per_prompt=4
#     → minimalne, żeby GRPO w ogóle policzył advantage
#     → cały run to ~50 kroków w ~30-45 min
#
#   eval_steps=10, eval_before_training=true
#     → eval zanim cokolwiek się nauczy = baseline
#     → drugi eval po 10 krokach = sprawdzenie czy eval
#       reward nie spada (co wskazywałoby że szum wciąż
#       przecieka na eval mimo is_eval fix)
#
#   ask_retriever_limit=1, retriever_search_limit=1
#     → K=1, S=1: jeden cykl komunikacji, jedno wyszukiwanie
#     → pipeline: search → retriever_message → ask → generate
#     → MUSI być >0, inaczej nie ma kanału do testowania
#
#   apply_length_penalty=true, lambda=0.005
#     → przy 200 tokenach kara=1.0, widoczna vs reward in [0,1]
#     → szukaj w W&B: ec/total_length_penalty > 0 od kroku 1
#
#   apply_channel_noise=true, noise_probability=0.15
#     → 15% tokenów zamaskowanych podczas treningu
#     → szukaj w W&B: ec/noise_applied=True w train logach
#       i ec/noise_applied=False w eval logach
#
#   max_steps=50 (przez num_episodes + mały dataset)
#     → wystarczy żeby zobaczyć czy metryki idą we właściwą stronę
#     → NIE wystarczy żeby nauczyć się kompresji (potrzeba ~500+ kroków)
# =================================================

srun --het-group=0 \
    $VENV_PYTHON -m marti.cli.multi_agent_train_ppo_ray \
    --pretrain "$TRAIN_MODEL" \
    --save_path "$OUTPUT_DIR" \
    --agents "$AGENT0" \
    --workflow_func_path "$WORKFLOW_SCRIPT" \
    --prompt_data "$TRAIN_DATA_PATH" \
    --eval_dataset "$EVAL_DATA_PATH" \
    --eval_split train \
    --eval_steps 10 \
    --eval_before_training \
    --eval_n_samples_per_prompt 1 \
    --workflow_args "{
        \"debug_dir\": \"$BASE_DIR/workflow_logs/$SLURM_JOB_ID\",
        \"embed_host\": \"$EMBED_NODE\",
        \"embed_port\": $EMBED_PORT,
        \"rerank_host\": \"$EMBED_NODE\",
        \"rerank_port\": $RERANK_PORT,
        \"similarity_weight\": 1.0,
        \"diversity_weight\": 1.0,
        \"groundedness_weight\": 0.0,
        \"relevancy_weight\": 0.0,
        \"weaviate_url\": \"http://$WEAVIATE_NODE_ID:8080\",
        \"debug\": \"true\",
        \"use_weaviate_context\": \"false\",
        \"weaviate_top_n\": 5,
        \"ask_retriever_limit\": 1,
        \"retriever_search_limit\": 1,
        \"apply_length_penalty\": \"true\",
        \"length_penalty_lambda\": 0.005,
        \"apply_channel_noise\": \"true\",
        \"noise_probability\": 0.15
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
    --train_batch_size 4 \
    --micro_train_batch_size 1 \
    --rollout_batch_size 4 \
    --n_samples_per_prompt 4 \
    --num_episodes 1 \
    --max_epochs 1 \
    --prompt_max_len 8096 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --gradient_checkpointing \
    --save_hf_ckpt \
    --seed 42 \
    --logging_steps 1 \
    --use_wandb "$WANDB_API_KEY" \
    --wandb_project MARTI_GRPO \
    --wandb_run_name "$WANDB_RUN"

echo "=== DIAGNOSTIC RUN COMPLETE ==="
echo ""
echo "Co sprawdzić w W&B i logach:"
echo "  1. ec_stats/mean_channel_tokens    — czy > 0 od kroku 1"
echo "  2. ec_stats/total_length_penalty   — czy > 0 od kroku 1 (powinno być ~1.0)"
echo "  3. eval reward krok 0 vs krok 10   — czy eval reward NIE spada"
echo "     (spadek = szum przecieka na eval, fix is_eval nie zadziałał)"
echo "  4. debug JSON w workflow_logs/      — czy ec_stats.channel_token_counts"
echo "     zawiera niezerowe wartości dla każdej trajektorii"
echo ""
echo "Komenda do szybkiego sprawdzenia debug logów:"
echo "  python3 -c \""
echo "  import json, glob, statistics"
echo "  files = glob.glob('$BASE_DIR/workflow_logs/$SLURM_JOB_ID/train/*.json')"
echo "  counts = [json.load(open(f))['ec_stats']['mean_channel_tokens'] for f in files[:20]]"
echo "  print('mean_channel_tokens — mean:', round(statistics.mean(counts),1),"
echo "        'min:', min(counts), 'max:', max(counts))"
echo "  \""

kill "$VLLM_EMBED_PID" 2>/dev/null
kill "$VLLM_RERANK_PID" 2>/dev/null