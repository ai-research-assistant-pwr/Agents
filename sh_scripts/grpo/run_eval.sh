#!/bin/bash
# =============================================================================
# run_eval.sh — Unified Evaluation Launcher (Multi-Folder Support)
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
shift # Przesuwamy argumenty - teraz $@ zawiera tylko listę folderów

if [ "$EXPERIMENT" != "exp4" ] && [ $# -eq 0 ]; then
    echo "ERROR: LOGS_DIR(s) not supplied."
    echo "Usage: sbatch run_eval.sh <EXPERIMENT> <LOGS_DIR_1> [LOGS_DIR_2 ...]"
    exit 1
fi

case "$EXPERIMENT" in
  exp1|exp2|exp3|all|exp2v2|exp3v2|allv2|exp4) ;;
  *)
    echo "ERROR: Unknown experiment '$EXPERIMENT'."
    exit 1
    ;;
esac

# Zapisujemy podane foldery do tablicy
LOGS_DIRS=("$@")

# =============================================================================
# 2. Configuration
# =============================================================================
MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"
VENV_PATH="$BASE_DIR/venv"

EMBED_MODEL="${EMBED_MODEL:-"Qwen/Qwen3-Embedding-4B"}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
NOISY_LOGS_DIR="${NOISY_LOGS_DIR:-}"
BASELINE_LOGS_DIR="${BASELINE_LOGS_DIR:-}"
SKIP_VLLM="${SKIP_VLLM:-false}"
EMBED_PORT="${EMBED_PORT:-8000}"

# Probe dirs (exp4)
PROBE_DIR_A="${PROBE_DIR_A:-$BASE_DIR/workflow_logs/noise_probe/eval_clean}"
PROBE_DIR_B="${PROBE_DIR_B:-$BASE_DIR/workflow_logs/noise_probe/eval_noisy}"
PROBE_LABEL_A="${PROBE_LABEL_A:-"clean (noise=0.0)"}"
PROBE_LABEL_B="${PROBE_LABEL_B:-"noisy"}"
PROBE_OUTPUT_SUFFIX="${PROBE_OUTPUT_SUFFIX:-"noise_probe"}"

source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0
source $VENV_PATH/bin/activate
VENV_PYTHON="$VENV_PATH/bin/python"

$VENV_PYTHON -m pip install hdbscan "huggingface_hub<1.0" > /dev/null 2>&1

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"
MY_NEW_TMP="$MY_DISK/tmp"
export XDG_CACHE_HOME="$MY_NEW_TMP/xdg_cache"
export TRITON_CACHE_DIR="$MY_NEW_TMP/triton_cache"
mkdir -p "$MY_NEW_TMP" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

# Ustawiamy, gdzie system (vLLM i HF) ma trzymać pobrane modele
export HF_HOME="$MY_NEW_TMP/huggingface"

EMBED_HOST="${EMBED_HOST:-localhost}"
VLLM_PID=""

# Helper: check if an experiment needs a vLLM server
_needs_vllm() {
    case "$1" in
        exp2|exp3|all|exp2v2|exp3v2|allv2|exp4) return 0 ;;
        *) return 1 ;;
    esac
}

# =============================================================================
# 3. Start vLLM Embedding Server (if needed) - URUCHAMIANY TYLKO RAZ
# =============================================================================
if _needs_vllm "$EXPERIMENT" && [ "$SKIP_VLLM" != "true" ]; then
    echo "=> Starting vLLM Embedding Server on localhost:$EMBED_PORT..."
    echo "   (If model is missing from HF_HOME, it will be downloaded now)"
    $VENV_PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$EMBED_MODEL" \
        --host 0.0.0.0 \
        --port $EMBED_PORT \
        --max-model-len 4096 &
    VLLM_PID=$!

    MAX_WAIT=300; WAITED=0  # Wydłużono czas oczekiwania do 5 minut na wypadek pobierania
    while ! curl -s http://localhost:$EMBED_PORT/v1/models > /dev/null 2>&1; do
        sleep 5; WAITED=$((WAITED+5))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: vLLM server did not start within 5 minutes. Aborting."
            [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null
            exit 1
        fi
    done
    echo "=> vLLM server online."
fi

# =============================================================================
# 4. Functions for running experiments
# =============================================================================
run_exp1() {
    local IN="$1"
    local OUT="$2/experiment_1"
    echo "    ─── Experiment 1: Morphology ────────────────────────────────"
    mkdir -p "$OUT"
    $VENV_PYTHON "$BASE_DIR/src/eval/eval_morphology.py" \
        --logs_dir "$IN" --output_dir "$OUT" --window_size "$WINDOW_SIZE"
}

run_exp2() {
    local IN="$1"
    local OUT="$2/experiment_2"
    echo "    ─── Experiment 2: Semantics (TopSim) ───────────────────────"
    mkdir -p "$OUT"
    EMBED_HOST="$EMBED_HOST" EMBED_PORT="$EMBED_PORT" EMBED_MODEL="$EMBED_MODEL" \
    $VENV_PYTHON "$BASE_DIR/src/eval/eval_semantics.py" \
        --logs_dir "$IN" --output_dir "$OUT" --window_size "$WINDOW_SIZE"
}

run_exp3() {
    local IN="$1"
    local OUT="$2/experiment_3"
    echo "    ─── Experiment 3: Pragmatics (Grounding) ───────────────────"
    mkdir -p "$OUT"
    NOISY_ARG=""; [ -n "$NOISY_LOGS_DIR" ] && NOISY_ARG="--noisy_logs_dir $NOISY_LOGS_DIR"
    $VENV_PYTHON "$BASE_DIR/src/eval/eval_pragmatics.py" \
        --logs_dir "$IN" --output_dir "$OUT" --window_size "$WINDOW_SIZE" \
        --embed_host "$EMBED_HOST" --embed_port "$EMBED_PORT" $NOISY_ARG
}

run_exp2v2() {
    local IN="$1"
    local OUT="$2/experiment_2"
    echo "    ─── Experiment 2v2: Semantics (Signal Structure) ───────────"
    mkdir -p "$OUT"
    BASELINE_ARG=""; [ -n "$BASELINE_LOGS_DIR" ] && BASELINE_ARG="--baseline_logs_dir $BASELINE_LOGS_DIR"
    EMBED_HOST="$EMBED_HOST" EMBED_PORT="$EMBED_PORT" EMBED_MODEL="$EMBED_MODEL" \
    $VENV_PYTHON "$BASE_DIR/src/eval/eval_semantics_v2.py" \
        --logs_dir "$IN" --output_dir "$OUT" --window_size "$WINDOW_SIZE" \
        --embed_host "$EMBED_HOST" --embed_port "$EMBED_PORT" $BASELINE_ARG
}

run_exp3v2() {
    local IN="$1"
    local OUT="$2/experiment_3"
    echo "    ─── Experiment 3v2: Pragmatics (Grounding v2) ──────────────"
    mkdir -p "$OUT"
    BASELINE_ARG=""; [ -n "$BASELINE_LOGS_DIR" ] && BASELINE_ARG="--baseline_logs_dir $BASELINE_LOGS_DIR"
    $VENV_PYTHON "$BASE_DIR/src/eval/eval_pragmatics_v2.py" \
        --logs_dir "$IN" --output_dir "$OUT" --window_size "$WINDOW_SIZE" \
        --embed_host "$EMBED_HOST" --embed_port "$EMBED_PORT" $BASELINE_ARG
}

run_exp4() {
    # Exp4 odpala się niezależnie od LOGS_DIRS
    echo "    ─── Experiment 4: Noise Probe (Generator Listening) ─────────"
    local OUT="$BASE_DIR/eval_results/$PROBE_OUTPUT_SUFFIX"
    mkdir -p "$OUT"
    $VENV_PYTHON "$BASE_DIR/src/eval/eval_noise_probe.py" \
        --dir_a "$PROBE_DIR_A" --dir_b "$PROBE_DIR_B" \
        --label_a "$PROBE_LABEL_A" --label_b "$PROBE_LABEL_B" \
        --output_dir "$OUT" --embed_host "$EMBED_HOST" \
        --embed_port "$EMBED_PORT" --embed_model "$EMBED_MODEL"
}


# =============================================================================
# 5. Main Execution Loop
# =============================================================================

if [ "$EXPERIMENT" = "exp4" ]; then
    echo "========================================================"
    echo " Processing: NOISE PROBE ($PROBE_OUTPUT_SUFFIX)"
    echo "========================================================"
    run_exp4
else
    for DIR_PATH in "${LOGS_DIRS[@]}"; do
        if [ ! -d "$DIR_PATH" ]; then
            echo "WARNING: Directory does not exist: $DIR_PATH. Skipping."
            continue
        fi

        REAL_PATH="$(realpath "$DIR_PATH")"
        RUN_NAME=$(basename "$REAL_PATH")
        CURRENT_OUT_DIR="$BASE_DIR/eval_results/$RUN_NAME"
        
        echo ""
        echo "========================================================"
        echo " Processing Directory : $RUN_NAME"
        echo " Path                 : $REAL_PATH"
        echo " Output               : $CURRENT_OUT_DIR"
        echo "========================================================"
        mkdir -p "$CURRENT_OUT_DIR"

        case "$EXPERIMENT" in
            exp1)   run_exp1   "$REAL_PATH" "$CURRENT_OUT_DIR" ;;
            exp2)   run_exp2   "$REAL_PATH" "$CURRENT_OUT_DIR" ;;
            exp3)   run_exp3   "$REAL_PATH" "$CURRENT_OUT_DIR" ;;
            exp2v2) run_exp2v2 "$REAL_PATH" "$CURRENT_OUT_DIR" ;;
            exp3v2) run_exp3v2 "$REAL_PATH" "$CURRENT_OUT_DIR" ;;
            all)
                run_exp1 "$REAL_PATH" "$CURRENT_OUT_DIR"
                run_exp2 "$REAL_PATH" "$CURRENT_OUT_DIR"
                run_exp3 "$REAL_PATH" "$CURRENT_OUT_DIR"
                ;;
            allv2)
                run_exp1   "$REAL_PATH" "$CURRENT_OUT_DIR"
                run_exp2v2 "$REAL_PATH" "$CURRENT_OUT_DIR"
                run_exp3v2 "$REAL_PATH" "$CURRENT_OUT_DIR"
                ;;
        esac

        # Manifest
        cat > "$CURRENT_OUT_DIR/eval_manifest.json" <<EOF
{
  "experiment":   "$EXPERIMENT",
  "run_name":     "$RUN_NAME",
  "logs_dir":     "$REAL_PATH",
  "output_dir":   "$CURRENT_OUT_DIR",
  "window_size":  $WINDOW_SIZE,
  "completed_at": "$(date -Iseconds)"
}
EOF
    done
fi

# =============================================================================
# 6. Cleanup
# =============================================================================
if [ -n "$VLLM_PID" ]; then
    kill "$VLLM_PID" 2>/dev/null
    echo "=> vLLM server stopped."
fi

echo ""
echo "========================================================"
echo " Evaluation pipeline complete."
echo "========================================================"