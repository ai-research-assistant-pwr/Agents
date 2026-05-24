#!/bin/bash
# =============================================================================
# run_eval.sh — Unified Evaluation Launcher
# =============================================================================
#
# Runs one or more evaluation experiments against a training run's workflow logs.
# Automatically starts (and stops) a vLLM embedding server for experiments that
# need embeddings (2, 3, 2v2, 3v2).
#
# Usage:
#   sbatch sh_scripts/grpo/eval/run_eval.sh <EXPERIMENT> <LOGS_DIR> [OPTIONS]
#
# Arguments:
#   EXPERIMENT   : exp1 | exp2 | exp3 | all
#                  exp2v2 | exp3v2 | allv2   ← v2 variants
#                  exp4                       ← noise probe (generator listening)
#
#   LOGS_DIR     : Path to the workflow_logs/<run_name> directory
#                  (for v2 experiments: this is the FULL / experimental run)
#                  (for exp4: not used, pass any existing dir e.g. ".")
#
# Options (env vars):
#   EMBED_MODEL          Model name for vLLM (default: Qwen/Qwen3-Embedding-4B)
#   EVAL_OUTPUT_DIR      Where to write results (default: Agents/eval_results/<run_name>)
#   WINDOW_SIZE          Window size for windowed analyses (default: 50)
#   SKIP_VLLM            Set to "true" if vLLM is already running externally
#   EMBED_HOST           Override embedding server host (default: localhost)
#   EMBED_PORT           Override embedding server port (default: 8000)
#
#   ── Original v1 options ──────────────────────────────────────────────────
#   NOISY_LOGS_DIR       (exp3 only) Path to noisy run logs for cross-run CIC
#
#   ── New v2 options ───────────────────────────────────────────────────────
#   BASELINE_LOGS_DIR    Path to the baseline run logs directory.
#                        Used by exp2v2 (comparison overlay) and
#                        exp3v2 (cross-run generator divergence, Probe 2).
#
#   ── Noise probe options (exp4) ───────────────────────────────────────────
#   PROBE_DIR_A          Folder z trajektoriami warunku A
#                        (default: $BASE_DIR/workflow_logs/noise_probe/eval_clean)
#   PROBE_DIR_B          Folder z trajektoriami warunku B
#                        (default: $BASE_DIR/workflow_logs/noise_probe/eval_noisy)
#   PROBE_LABEL_A        Etykieta A na wykresach (default: "clean (noise=0.0)")
#   PROBE_LABEL_B        Etykieta B na wykresach (default: "noisy")
#   PROBE_OUTPUT_SUFFIX  Suffix dołączany do output_dir (default: "noise_probe")
#
# Examples:
#   # Porównanie clean vs noisy (0.5) z etykietami:
#   PROBE_DIR_A=Agents/workflow_logs/noise_probe/eval_clean \
#   PROBE_DIR_B=Agents/workflow_logs/noise_probe/eval_noisy_05 \
#   PROBE_LABEL_A="clean (noise=0.0)" \
#   PROBE_LABEL_B="noisy (noise=0.5)" \
#   PROBE_OUTPUT_SUFFIX="clean_vs_05" \
#   sbatch sh_scripts/grpo/eval/run_eval.sh exp4 .
#
#   # Porównanie noisy 0.5 vs noisy 1.0:
#   PROBE_DIR_A=Agents/workflow_logs/noise_probe/eval_noisy_05 \
#   PROBE_DIR_B=Agents/workflow_logs/noise_probe/eval_noisy_10 \
#   PROBE_LABEL_A="noisy (noise=0.5)" \
#   PROBE_LABEL_B="noisy (noise=1.0)" \
#   PROBE_OUTPUT_SUFFIX="05_vs_10" \
#   sbatch sh_scripts/grpo/eval/run_eval.sh exp4 .
#
# Examples:
#   # Run all v2 experiments (full run + baseline comparison):
#   BASELINE_LOGS_DIR=Agents/workflow_logs/baseline_20250512 \
#   sbatch sh_scripts/grpo/eval/run_eval.sh allv2 \
#       Agents/workflow_logs/exp_full_20250512
#
#   # Run only exp2v2 (signal structure, no baseline):
#   sbatch sh_scripts/grpo/eval/run_eval.sh exp2v2 \
#       Agents/workflow_logs/exp_full_20250512
#
#   # Run exp3v2 with baseline for cross-run generator divergence:
#   BASELINE_LOGS_DIR=Agents/workflow_logs/baseline_20250512 \
#   sbatch sh_scripts/grpo/eval/run_eval.sh exp3v2 \
#       Agents/workflow_logs/exp_full_20250512
#
#   # Run only experiment 1 (no embedding server needed):
#   sbatch sh_scripts/grpo/eval/run_eval.sh exp1 \
#       Agents/workflow_logs/baseline_free_20250512_120000
#
#   # Run original experiment 3 with cross-run CIC (v1):
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

# exp4 nie potrzebuje LOGS_DIR — pozwalamy przekazać "." lub dowolny istniejący katalog
if [ "$EXPERIMENT" != "exp4" ] && [ ! -d "$LOGS_DIR" ]; then
    echo "ERROR: LOGS_DIR does not exist: $LOGS_DIR"
    exit 1
fi

case "$EXPERIMENT" in
  exp1|exp2|exp3|all|exp2v2|exp3v2|allv2|exp4) ;;
  *)
    echo "ERROR: Unknown experiment '$EXPERIMENT'."
    echo "       Valid: exp1 | exp2 | exp3 | all | exp2v2 | exp3v2 | allv2 | exp4"
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
BASELINE_LOGS_DIR="${BASELINE_LOGS_DIR:-}"
SKIP_VLLM="${SKIP_VLLM:-false}"
EMBED_PORT="${EMBED_PORT:-8000}"

# Noise probe (exp4) — zmienne z domyślnym katalogiem noise_probe
PROBE_DIR_A="${PROBE_DIR_A:-$BASE_DIR/workflow_logs/noise_probe/eval_clean}"
PROBE_DIR_B="${PROBE_DIR_B:-$BASE_DIR/workflow_logs/noise_probe/eval_noisy}"
PROBE_LABEL_A="${PROBE_LABEL_A:-"clean (noise=0.0)"}"
PROBE_LABEL_B="${PROBE_LABEL_B:-"noisy"}"
PROBE_OUTPUT_SUFFIX="${PROBE_OUTPUT_SUFFIX:-"noise_probe"}"

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

$VENV_PYTHON -m pip install hdbscan

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
[ -n "$BASELINE_LOGS_DIR" ] && echo " Baseline logs    : $BASELINE_LOGS_DIR"
[ -n "$NOISY_LOGS_DIR"    ] && echo " Noisy logs (v1)  : $NOISY_LOGS_DIR"
echo "========================================================"

# =============================================================================
# Helper: check if an experiment needs a vLLM server
# =============================================================================
_needs_vllm() {
    case "$1" in
        exp2|exp3|all|exp2v2|exp3v2|allv2|exp4) return 0 ;;
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

run_exp2v2() {
    echo ""
    echo "─── Experiment 2v2: Semantics (Signal Structure) ───────────"
    local OUT="$EVAL_OUTPUT_DIR/experiment_2"
    mkdir -p "$OUT"

    BASELINE_ARG=""
    if [ -n "$BASELINE_LOGS_DIR" ]; then
        BASELINE_ARG="--baseline_logs_dir $BASELINE_LOGS_DIR"
        echo "    Baseline comparison: $BASELINE_LOGS_DIR"
    fi

    EMBED_HOST="$EMBED_HOST" EMBED_PORT="$EMBED_PORT" EMBED_MODEL="$EMBED_MODEL" \
    $VENV_PYTHON "$BASE_DIR/src/eval/eval_semantics_v2.py" \
        --logs_dir   "$LOGS_DIR" \
        --output_dir "$OUT" \
        --window_size "$WINDOW_SIZE" \
        --embed_host  "$EMBED_HOST" \
        --embed_port  "$EMBED_PORT" \
        $BASELINE_ARG
    echo "    Exp 2v2 done → $OUT"
}

run_exp3v2() {
    echo ""
    echo "─── Experiment 3v2: Pragmatics (Grounding v2) ──────────────"
    local OUT="$EVAL_OUTPUT_DIR/experiment_3"
    mkdir -p "$OUT"

    BASELINE_ARG=""
    if [ -n "$BASELINE_LOGS_DIR" ]; then
        BASELINE_ARG="--baseline_logs_dir $BASELINE_LOGS_DIR"
        echo "    Cross-run divergence: $BASELINE_LOGS_DIR"
    fi

    $VENV_PYTHON "$BASE_DIR/src/eval/eval_pragmatics_v2.py" \
        --logs_dir    "$LOGS_DIR" \
        --output_dir  "$OUT" \
        --window_size "$WINDOW_SIZE" \
        --embed_host  "$EMBED_HOST" \
        --embed_port  "$EMBED_PORT" \
        $BASELINE_ARG
    echo "    Exp 3v2 done → $OUT"
}

run_exp4() {
    echo ""
    echo "─── Experiment 4: Noise Probe (Generator Listening) ─────────"

    local OUT="$EVAL_OUTPUT_DIR/$PROBE_OUTPUT_SUFFIX"
    mkdir -p "$OUT"

    echo "    Dir A  : $PROBE_DIR_A  (${PROBE_LABEL_A})"
    echo "    Dir B  : $PROBE_DIR_B  (${PROBE_LABEL_B})"
    echo "    Output : $OUT"

    if [ ! -d "$PROBE_DIR_A" ]; then
        echo "ERROR: PROBE_DIR_A nie istnieje: $PROBE_DIR_A"
        echo "       Ustaw zmienną PROBE_DIR_A lub uruchom run_eval_clean.sh."
        return 1
    fi
    if [ ! -d "$PROBE_DIR_B" ]; then
        echo "ERROR: PROBE_DIR_B nie istnieje: $PROBE_DIR_B"
        echo "       Ustaw zmienną PROBE_DIR_B lub uruchom run_eval_noisy.sh."
        return 1
    fi

    $VENV_PYTHON "$BASE_DIR/src/eval/eval_noise_probe.py" \
        --dir_a       "$PROBE_DIR_A" \
        --dir_b       "$PROBE_DIR_B" \
        --label_a     "$PROBE_LABEL_A" \
        --label_b     "$PROBE_LABEL_B" \
        --output_dir  "$OUT" \
        --embed_host  "$EMBED_HOST" \
        --embed_port  "$EMBED_PORT" \
        --embed_model "$EMBED_MODEL"

    echo "    Exp 4 done → $OUT"
}

case "$EXPERIMENT" in
    exp1)   run_exp1 ;;
    exp2)   run_exp2 ;;
    exp3)   run_exp3 ;;
    exp4)   run_exp4 ;;
    all)
        run_exp1
        run_exp2
        run_exp3
        ;;
    exp2v2) run_exp2v2 ;;
    exp3v2) run_exp3v2 ;;
    allv2)
        run_exp1
        run_exp2v2
        run_exp3v2
        ;;
esac

# =============================================================================
# 5. Write consolidated eval manifest
# =============================================================================
cat > "$EVAL_OUTPUT_DIR/eval_manifest.json" <<EOF
{
  "experiment":         "$EXPERIMENT",
  "run_name":           "$RUN_NAME",
  "logs_dir":           "$LOGS_DIR",
  "baseline_logs_dir":  "$BASELINE_LOGS_DIR",
  "probe_dir_a":        "$PROBE_DIR_A",
  "probe_dir_b":        "$PROBE_DIR_B",
  "probe_label_a":      "$PROBE_LABEL_A",
  "probe_label_b":      "$PROBE_LABEL_B",
  "output_dir":         "$EVAL_OUTPUT_DIR",
  "window_size":        $WINDOW_SIZE,
  "slurm_job_id":       "$SLURM_JOB_ID",
  "completed_at":       "$(date -Iseconds)"
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