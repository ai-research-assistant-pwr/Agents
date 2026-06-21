#!/usr/bin/env bash
###############################################################################
# Embedding-mode launcher: serves TWO vLLM models on ONE GPU.
#
#   Single GPU -> vLLM embedding model  (EMB_PORT)   [EMB_FRACTION of VRAM]
#              -> vLLM chat model       (CHAT_PORT)  [1-EMB_FRACTION of VRAM]
#
# GPU memory is split by EMB_FRACTION; safety margin applied to free VRAM.
#
# Usage:
#   ./sh_scripts/serve_vllm.sh     # start all servers in background
#   ./sh_scripts/stop.sh           # stop all servers
###############################################################################
set -uo pipefail

# =============================================================================
# CONFIGURATION — edit here, not below
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_ROOT}/config/app/config.yaml"

# Optional: set to /path/to/venv/bin/activate to use a virtual environment.
VENV_PATH=""

ROOT_PATH="${PROJECT_ROOT}"
MODELS_DIR="models"
MODELS_PATH="${ROOT_PATH}/${MODELS_DIR}"

# Chat model — path and port are read from config/app/config.yaml
CHAT_MODEL_PATH=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['generator']['weights'])")
CHAT_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['generator']['port'])")
CHAT_SERVED_NAME="Qwen/Qwen3-4B"
CHAT_MODEL_LEN=16384
CHAT_MAX_SEQS=2

# Embedding model
EMB_MODEL="Qwen/Qwen3-Embedding-4B"
EMB_PORT=8018
EMB_MODEL_LEN=512

GPU_INDEX=0   # physical GPU index (PCI bus order) to use for both models

# Fraction of GPU's available VRAM reserved for the embedder.
# Chat model gets the remaining (1 - EMB_FRACTION).
EMB_FRACTION=0.30

# Safety margin applied to free VRAM when computing utilization fractions.
SAFETY_MARGIN=0.90

# Abort early if free VRAM is below this threshold (MiB).
MIN_FREE_MIB=4000

# PID / log directory
TMP_DIR="${ROOT_PATH}/tmp"

# Compile cache directories (deleted at startup to avoid stale artefacts)
COMPILE_CACHE_DIR="${MODELS_PATH}/compile_cache_emb"
COMPILE_CACHE_DIR_2="${MODELS_PATH}/compile_cache_chat"

# ---------------------------------------------------------------------------
# Sanity checks — all required variables must be set
# ---------------------------------------------------------------------------
: "${ROOT_PATH:?Set ROOT_PATH}"
: "${MODELS_DIR:?Set MODELS_DIR}"
: "${EMB_MODEL:?Set EMB_MODEL}"
: "${EMB_PORT:?Set EMB_PORT}"
: "${EMB_MODEL_LEN:?Set EMB_MODEL_LEN}"
: "${CHAT_PORT:?Set CHAT_PORT}"
: "${CHAT_MODEL_LEN:?Set CHAT_MODEL_LEN}"
: "${CHAT_MODEL_PATH:?CHAT_MODEL_PATH not found in config}"

# ---------------------------------------------------------------------------
# Activate venv (if configured)
# ---------------------------------------------------------------------------
[ -n "${VENV_PATH}" ] && source "${VENV_PATH}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ---------------------------------------------------------------------------
# Cache environment variables
# ---------------------------------------------------------------------------
export TORCHINDUCTOR_CACHE_DIR="${ROOT_PATH}/.cache/torch_inductor"
export VLLM_CACHE_ROOT="${ROOT_PATH}/.cache/vllm"
export HF_HOME="${ROOT_PATH}/.cache/hf"
export TRITON_CACHE_DIR="${ROOT_PATH}/.cache/triton"
export FLASHINFER_CACHE_DIR="${ROOT_PATH}/.cache/flashinfer"
export TORCH_EXTENSIONS_DIR="${ROOT_PATH}/.cache/torch_extensions"

# ---------------------------------------------------------------------------
# Set up directories; delete compile caches so we start clean each time
# ---------------------------------------------------------------------------
mkdir -p "${MODELS_PATH}" "${TMP_DIR}"
rm -rf "${COMPILE_CACHE_DIR}";   mkdir -p "${COMPILE_CACHE_DIR}"
rm -rf "${COMPILE_CACHE_DIR_2}"; mkdir -p "${COMPILE_CACHE_DIR_2}"

# ---------------------------------------------------------------------------
# Prometheus patch — do NOT remove.
# Fixes _IncludedRouter compatibility bug upstream in prometheus-fastapi-instrumentator.
# ---------------------------------------------------------------------------
ROUTING_FILE=$(python3 -c \
    "import prometheus_fastapi_instrumentator.routing as r, inspect; print(inspect.getfile(r))" \
    2>/dev/null || true)
if [ -n "${ROUTING_FILE}" ]; then
    sed -i \
      's/            route_name = route\.path/            if not hasattr(route, "path"):\n                continue\n            route_name = route.path/' \
      "${ROUTING_FILE}" 2>/dev/null || true
    echo "Patched ${ROUTING_FILE}"
fi

# ---------------------------------------------------------------------------
# Helper: query a field from a specific physical GPU by index
# ---------------------------------------------------------------------------
gpu_query() {   # $1 = field (e.g. memory.total), $2 = gpu index
    nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits -i "$2" | tr -d ' '
}

# ---------------------------------------------------------------------------
# GPU memory budget — split between embedder and chat model
# ---------------------------------------------------------------------------
GPU_TOTAL_MEM=$(gpu_query memory.total "${GPU_INDEX}")
GPU_USED_MEM=$(gpu_query memory.used  "${GPU_INDEX}")
GPU_FREE_MEM=$(gpu_query memory.free  "${GPU_INDEX}")
echo "[GPU ${GPU_INDEX}] total: ${GPU_TOTAL_MEM} MiB | used: ${GPU_USED_MEM} MiB | free: ${GPU_FREE_MEM} MiB"

if [ "${GPU_FREE_MEM}" -lt "${MIN_FREE_MIB}" ]; then
    echo "ERROR: GPU ${GPU_INDEX} only ${GPU_FREE_MEM} MiB free, need >= ${MIN_FREE_MIB} MiB. Aborting."
    exit 1
fi

AVAILABLE_MIB=$(awk "BEGIN {printf \"%.0f\", ${GPU_FREE_MEM} * ${SAFETY_MARGIN}}")

EMB_BUDGET_MIB=$(awk  "BEGIN {printf \"%.0f\", ${AVAILABLE_MIB} * ${EMB_FRACTION}}")
CHAT_BUDGET_MIB=$(awk "BEGIN {printf \"%.0f\", ${AVAILABLE_MIB} * (1 - ${EMB_FRACTION})}")

EMB_VLLM_UTILIZATION=$(awk  "BEGIN {printf \"%.4f\", ${EMB_BUDGET_MIB}  / ${GPU_TOTAL_MEM}}")
CHAT_VLLM_UTILIZATION=$(awk "BEGIN {printf \"%.4f\", ${CHAT_BUDGET_MIB} / ${GPU_TOTAL_MEM}}")

EMB_PERCENT=$(awk "BEGIN {printf \"%d\", ${EMB_FRACTION} * 100}")
CHAT_PERCENT=$((100 - EMB_PERCENT))

echo "[GPU ${GPU_INDEX}] embedder budget: ${EMB_BUDGET_MIB} MiB (${EMB_PERCENT}%) | chat budget: ${CHAT_BUDGET_MIB} MiB (${CHAT_PERCENT}%)"
echo "embedder gpu-memory-utilization: ${EMB_VLLM_UTILIZATION} | chat gpu-memory-utilization: ${CHAT_VLLM_UTILIZATION}"

# ---------------------------------------------------------------------------
# Launch vLLM EMBEDDING model
# ---------------------------------------------------------------------------
echo "Starting ${EMB_MODEL} on GPU ${GPU_INDEX} (port ${EMB_PORT})..."

env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
vllm serve "${EMB_MODEL}" \
    --port "${EMB_PORT}" \
    --tensor-parallel-size 1 \
    --download-dir "${MODELS_PATH}" \
    --compilation-config "{\"cache_dir\": \"${COMPILE_CACHE_DIR}\"}" \
    --gpu-memory-utilization "${EMB_VLLM_UTILIZATION}" \
    --runner "pooling" \
    --dtype "bfloat16" \
    --max-model-len "${EMB_MODEL_LEN}" \
    --host 0.0.0.0 \
    >> "${TMP_DIR}/vllm_embedding.log" 2>&1 &

EMB_PID=$!
disown "${EMB_PID}"
echo "${EMB_PID}" > "${TMP_DIR}/vllm_embedding.pid"
echo "Embedder PID ${EMB_PID} → ${TMP_DIR}/vllm_embedding.pid"

echo "Waiting for ${EMB_MODEL} to load..."
while [ "$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${EMB_PORT}/health")" != "200" ]; do
    if ! kill -0 "${EMB_PID}" 2>/dev/null; then
        echo "ERROR: Embedder process (PID ${EMB_PID}) exited unexpectedly."
        echo "       Check logs: tail -50 ${TMP_DIR}/vllm_embedding.log"
        exit 1
    fi
    sleep 10
    echo "  Still loading ${EMB_MODEL}..."
done
echo "Embedding model ready!"

# ---------------------------------------------------------------------------
# Launch vLLM CHAT model
# ---------------------------------------------------------------------------
echo "Starting ${CHAT_MODEL_PATH} on GPU ${GPU_INDEX} (port ${CHAT_PORT})..."

env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
vllm serve "${CHAT_MODEL_PATH}" \
    --served-model-name "${CHAT_SERVED_NAME}" \
    --port "${CHAT_PORT}" \
    --tensor-parallel-size 1 \
    --download-dir "${MODELS_PATH}" \
    --compilation-config "{\"cache_dir\": \"${COMPILE_CACHE_DIR_2}\"}" \
    --gpu-memory-utilization "${CHAT_VLLM_UTILIZATION}" \
    --dtype "bfloat16" \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --max-model-len "${CHAT_MODEL_LEN}" \
    --max-num-seqs "${CHAT_MAX_SEQS}" \
    --host 0.0.0.0 \
    >> "${TMP_DIR}/vllm_chat.log" 2>&1 &

CHAT_PID=$!
disown "${CHAT_PID}"
echo "${CHAT_PID}" > "${TMP_DIR}/vllm_chat.pid"
echo "Chat model PID ${CHAT_PID} → ${TMP_DIR}/vllm_chat.pid"

echo "Waiting for ${CHAT_MODEL_PATH} to load..."
while [ "$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${CHAT_PORT}/health")" != "200" ]; do
    if ! kill -0 "${CHAT_PID}" 2>/dev/null; then
        echo "ERROR: Chat model process (PID ${CHAT_PID}) exited unexpectedly."
        echo "       Check logs: tail -50 ${TMP_DIR}/vllm_chat.log"
        exit 1
    fi
    sleep 10
    echo "  Still loading chat model..."
done
echo "Chat model ready!"

# ---------------------------------------------------------------------------
# Monitor loop (background) — logs health + GPU usage every 60 s
# ---------------------------------------------------------------------------
(
    while true; do
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        EMB_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${EMB_PORT}/health")
        CHAT_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${CHAT_PORT}/health")
        GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_INDEX}" | tr -d ' ')
        GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_INDEX}" | tr -d ' ')

        echo "[${TIMESTAMP}] emb:${EMB_STATUS} chat:${CHAT_STATUS} | GPU${GPU_INDEX} used:${GPU_USED} free:${GPU_FREE}"
        [ "${EMB_STATUS}"  != "200" ] && echo "[${TIMESTAMP}] WARNING: embedding unhealthy (HTTP ${EMB_STATUS})"
        [ "${CHAT_STATUS}" != "200" ] && echo "[${TIMESTAMP}] WARNING: chat unhealthy (HTTP ${CHAT_STATUS})"
        sleep 60
    done
) >> "${TMP_DIR}/vllm_monitor.log" 2>&1 &

MON_PID=$!
disown "${MON_PID}"
echo "${MON_PID}" > "${TMP_DIR}/vllm_monitor.pid"
echo "Monitor PID ${MON_PID} — tail -f ${TMP_DIR}/vllm_monitor.log"

# ---------------------------------------------------------------------------
# Done — servers are running detached; script exits now
# ---------------------------------------------------------------------------
echo ""
echo "All servers running in background."
echo "  Embedding : http://localhost:${EMB_PORT}   (PID ${EMB_PID})"
echo "  Chat      : http://localhost:${CHAT_PORT}  (PID ${CHAT_PID})"
echo ""
echo "Logs:"
echo "  tail -f ${TMP_DIR}/vllm_embedding.log"
echo "  tail -f ${TMP_DIR}/vllm_chat.log"
echo "  tail -f ${TMP_DIR}/vllm_monitor.log"
echo ""
echo "Stop all servers: ./sh_scripts/stop.sh"
