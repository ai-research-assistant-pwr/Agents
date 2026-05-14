#!/bin/bash
# =============================================================================
# run_eval.sh — Unified Evaluation Launcher
# =============================================================================
#
# Runs one or more evaluation experiments against a training run's workflow logs.
# Automatically starts (and stops) a vLLM embedding server for experiments that
# need embeddings (2 and 3).
#
# Usage:
#   sbatch sh_scripts/grpo/eval/run_eval.sh <EXPERIMENT> <LOGS_DIR> [OPTIONS]
#
# Arguments:
#   EXPERIMENT   : exp1 | exp2 | exp3 | all
#   LOGS_DIR     : Path to the workflow_logs/<run_name> directory
#
# Options (env vars):
#   EMBED_MODEL          Model name for vLLM (default: Qwen/Qwen3-Embedding-4B)
#   EVAL_OUTPUT_DIR      Where to write results (default: Agents/eval_results/<run_name>)
#   WINDOW_SIZE          Window size for windowed analyses (default: 50)
#   NOISY_LOGS_DIR       (exp3 only) Path to noisy run logs for cross-run CIC
#   SKIP_VLLM            Set to "true" if vLLM is already running externally
#   EMBED_HOST           Override embedding server host (default: localhost)
#   EMBED_PORT           Override embedding server port (default: 8000)
#
# Examples:
#   # Run all experiments on the last exp_full training run:
#   sbatch sh_scripts/grpo/eval/run_eval.sh all \
#       Agents/workflow_logs/exp_full_20250512_143000
#
#   # Run only experiment 1 (no embedding server needed):
#   sbatch sh_scripts/grpo/eval/run_eval.sh exp1 \
#       Agents/workflow_logs/baseline_free_20250512_120000
#
#   # Run experiment 3 with cross-run CIC:
#   NOISY_LOGS_DIR=Agents/workflow_logs/exp_full_20250512_143000 \
#   sbatch sh_scripts/grpo/eval/run_eval.sh exp3 \
#       Agents/workflow_logs/baseline_free_20250512_120000
#
# =============================================================================
#SBATCH --job-name=eval_emergent
#SBATCH --output=Agents/out/%x_%j.out
#SBATCH --time=0-02:00:00
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:hopper:1
#SBATCH --ntasks-per-node=1

set -e

# =============================================================================
# 1. Arguments
# =============================================================================
EXPERIMENT="${1:-all}"
LOGS_DIR="${2:-}"

if [ -z "$LOGS_DIR" ]; then
    echo "ERROR: LOGS_DIR not supplied."
    echo "Usage: sbatch run_eval.sh <EXPERIMENT> <LOGS_DIR>"
    exit 1
fi

if [ ! -d "$LOGS_DIR" ]; then
    echo "ERROR: LOGS_DIR does not exist: $LOGS_DIR"
    exit 1
fi

case "$EXPERIMENT" in
  exp1|exp2|exp3|all) ;;
  *)
    echo "ERROR: Unknown experiment '$EXPERIMENT'. Valid: exp1 | exp2 | exp3 | all"
    exit 1
    ;;
esac

# =============================================================================
# 2. Configuration
# =============================================================================
MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"
VENV_PATH="$BASE_DIR/venv"

EMBED_MODEL="${EMBED_MODEL:-"Qwen/Qwen3-Embedding-4B"}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
NOISY_LOGS_DIR="${NOISY_LOGS_DIR:-}"
SKIP_VLLM="${SKIP_VLLM:-false}"
EMBED_PORT="${EMBED_PORT:-8000}"

# Derive output dir from logs dir basename
RUN_NAME=$(basename "$LOGS_DIR")
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$BASE_DIR/eval_results/$RUN_NAME}"

mkdir -p "$EVAL_OUTPUT_DIR"

# Resolve absolute path for LOGS_DIR
LOGS_DIR="$(realpath "$LOGS_DIR")"

source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0
source $VENV_PATH/bin/activate
VENV_PYTHON="$VENV_PATH/bin/python"

echo "=> Sprawdzanie i instalacja scikit-learn..."
pip install -q scikit-learn

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

MY_NEW_TMP="$MY_DISK/tmp"
export XDG_CACHE_HOME="$MY_NEW_TMP/xdg_cache"
export TRITON_CACHE_DIR="$MY_NEW_TMP/triton_cache"
mkdir -p "$MY_NEW_TMP" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

EMBED_HOST="${EMBED_HOST:-localhost}"
VLLM_PID=""

echo "========================================================"
echo " Eval experiment  : $EXPERIMENT"
echo " Logs dir         : $LOGS_DIR"
echo " Output dir       : $EVAL_OUTPUT_DIR"
echo " Window size      : $WINDOW_SIZE"
echo "========================================================"

# =============================================================================
# Helper: check if an experiment needs a vLLM server
# =============================================================================
_needs_vllm() {
    case "$1" in
        exp2|exp3|all) return 0 ;;
        *) return 1 ;;
    esac
}

# =============================================================================
# 3. Start vLLM Embedding Server (if needed)
# =============================================================================
if _needs_vllm "$EXPERIMENT" && [ "$SKIP_VLLM" != "true" ]; then
    echo "=> Starting vLLM Embedding Server on localhost:$EMBED_PORT..."
    $VENV_PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$EMBED_MODEL" \
        --host 0.0.0.0 \
        --port $EMBED_PORT \
        --max-model-len 4096 &
    VLLM_PID=$!

    MAX_WAIT=120; WAITED=0
    while ! curl -s http://localhost:$EMBED_PORT/v1/models > /dev/null 2>&1; do
        sleep 5; WAITED=$((WAITED+5))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: vLLM server did not start. Aborting."
            [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null
            exit 1
        fi
    done
    echo "=> vLLM server online."
fi

# =============================================================================
# 4. Run Experiments
# =============================================================================

run_exp1() {
    echo ""
    echo "─── Experiment 1: Morphology ────────────────────────────────"
    local OUT="$EVAL_OUTPUT_DIR/experiment_1"
    mkdir -p "$OUT"
    $VENV_PYTHON "$BASE_DIR/src/eval/eval_morphology.py" \
        --logs_dir   "$LOGS_DIR" \
        --output_dir "$OUT" \
        --window_size "$WINDOW_SIZE"
    echo "    Exp 1 done → $OUT"
}

run_exp2() {
    echo ""
    echo "─── Experiment 2: Semantics (TopSim) ───────────────────────"
    local OUT="$EVAL_OUTPUT_DIR/experiment_2"
    mkdir -p "$OUT"
    EMBED_HOST="$EMBED_HOST" EMBED_PORT="$EMBED_PORT" EMBED_MODEL="$EMBED_MODEL" \
    $VENV_PYTHON "$BASE_DIR/src/eval/eval_semantics.py" \
        --logs_dir   "$LOGS_DIR" \
        --output_dir "$OUT" \
        --window_size "$WINDOW_SIZE"
    echo "    Exp 2 done → $OUT"
}

run_exp3() {
    echo ""
    echo "─── Experiment 3: Pragmatics (Grounding) ───────────────────"
    local OUT="$EVAL_OUTPUT_DIR/experiment_3"
    mkdir -p "$OUT"

    NOISY_ARG=""
    if [ -n "$NOISY_LOGS_DIR" ]; then
        NOISY_ARG="--noisy_logs_dir $NOISY_LOGS_DIR"
        echo "    Cross-run CIC: $NOISY_LOGS_DIR"
    fi

    $VENV_PYTHON "$BASE_DIR/src/eval/eval_pragmatics.py" \
        --logs_dir    "$LOGS_DIR" \
        --output_dir  "$OUT" \
        --window_size "$WINDOW_SIZE" \
        --embed_host  "$EMBED_HOST" \
        --embed_port  "$EMBED_PORT" \
        $NOISY_ARG
    echo "    Exp 3 done → $OUT"
}

case "$EXPERIMENT" in
    exp1) run_exp1 ;;
    exp2) run_exp2 ;;
    exp3) run_exp3 ;;
    all)
        run_exp1
        run_exp2
        run_exp3
        ;;
esac

# =============================================================================
# 5. Write consolidated eval manifest
# =============================================================================
cat > "$EVAL_OUTPUT_DIR/eval_manifest.json" <<EOF
{
  "experiment":    "$EXPERIMENT",
  "run_name":      "$RUN_NAME",
  "logs_dir":      "$LOGS_DIR",
  "output_dir":    "$EVAL_OUTPUT_DIR",
  "window_size":   $WINDOW_SIZE,
  "slurm_job_id":  "$SLURM_JOB_ID",
  "completed_at":  "$(date -Iseconds)"
}
EOF

# =============================================================================
# 6. Cleanup
# =============================================================================
if [ -n "$VLLM_PID" ]; then
    kill "$VLLM_PID" 2>/dev/null
    echo "=> vLLM server stopped."
fi

echo ""
echo "========================================================"
echo " Evaluation complete."
echo " Results: $EVAL_OUTPUT_DIR"
echo "========================================================"