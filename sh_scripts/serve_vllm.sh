#!/usr/bin/env bash
# Start vLLM for the locally fine-tuned Qwen3-4B model and the embedding model.
#
# Reads model path and port from config/app/config.yaml, then either:
#   (default) launches vLLM via Docker Compose
#   --no-docker  runs `vllm serve` directly (vllm must be installed)
#
# Usage:
#   ./sh_scripts/serve_vllm.sh            # Docker mode
#   ./sh_scripts/serve_vllm.sh --no-docker

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_ROOT/config/app/config.yaml"

USE_DOCKER=true
for arg in "$@"; do
  [[ "$arg" == "--no-docker" ]] && USE_DOCKER=false
done

# Parse config/app/config.yaml — python3 is already required by the project
WEIGHTS=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['generator']['weights'])")
PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['generator']['port'])")
EMBED_MODEL="Qwen/Qwen3-Embedding-4B"
EMBED_PORT=8005

echo "Model weights : $WEIGHTS"
echo "Port          : $PORT"
echo "Embed model   : $EMBED_MODEL"
echo "Embed port    : $EMBED_PORT"
echo ""

if $USE_DOCKER; then
  # Convert the host-relative weights path (models/…) to the container path (/models/…).
  # The docker-compose volume mounts {project_root}/models as /models inside the container.
  CONTAINER_MODEL="/${WEIGHTS%/}"   # "models/foo/bar" → "/models/foo/bar"
  export VLLM_MODEL="$CONTAINER_MODEL"
  export VLLM_PORT="$PORT"
  export VLLM_EMBED_MODEL="$EMBED_MODEL"
  export VLLM_EMBED_PORT="$EMBED_PORT"
  echo "Starting vLLM via Docker Compose (VLLM_MODEL=$VLLM_MODEL, VLLM_PORT=$VLLM_PORT)"
  echo "Starting embedding model via Docker Compose (VLLM_EMBED_MODEL=$VLLM_EMBED_MODEL, VLLM_EMBED_PORT=$VLLM_EMBED_PORT)"
  docker compose -f "$PROJECT_ROOT/docker/vllm/docker-compose.yml" up
else
  #HOST_MODEL="$PROJECT_ROOT/$WEIGHTS"
  HOST_MODEL=$WEIGHTS
  cleanup() {
    local pids=()
    [[ -n "${GEN_PID:-}" ]] && pids+=("$GEN_PID")
    [[ -n "${EMBED_PID:-}" ]] && pids+=("$EMBED_PID")
    ((${#pids[@]})) && kill "${pids[@]}" 2>/dev/null || true
  }
  trap cleanup EXIT

  # Patch prometheus-fastapi-instrumentator for FastAPI _IncludedRouter compatibility.
  # See: https://github.com/vllm-project/vllm/issues — upstream bug, not fixed in the package.
  ROUTING_FILE=$(python3 -c \
    "import prometheus_fastapi_instrumentator.routing as r, inspect; print(inspect.getfile(r))" \
    2>/dev/null || true)
  if [ -n "$ROUTING_FILE" ]; then
    sed -i \
      's/            route_name = route\.path/            if not hasattr(route, "path"):\n                continue\n            route_name = route.path/' \
      "$ROUTING_FILE" 2>/dev/null || true
    echo "Patched $ROUTING_FILE"
  fi

  echo "Starting vLLM directly (model=$HOST_MODEL)"
  PYTHONUNBUFFERED=1 vllm serve "$HOST_MODEL" \
      --served-model-name Qwen/Qwen3-4B \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.5 \
      --max-model-len 8192 \
      --max-num-seqs 4 \
      --enforce-eager \
      --reasoning-parser qwen3 \
      --port "$PORT" \
      --host 0.0.0.0 &
  GEN_PID=$!

  echo "Starting embedding model directly (model=$EMBED_MODEL)"
  PYTHONUNBUFFERED=1 python -m vllm.entrypoints.openai.api_server \
      --model "$EMBED_MODEL" \
      --port "$EMBED_PORT" \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.4 \
      --host 0.0.0.0 &
  EMBED_PID=$!

  wait_for_server() {
    local pid="$1"
    local url="$2"
    local label="$3"
    while ! curl -sf "$url" >/dev/null 2>&1; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "$label exited before it became ready" >&2
        wait "$pid" || true
        exit 1
      fi
      sleep 2
    done
  }

  echo "Waiting for direct vLLM servers to become ready..."
  wait_for_server "$GEN_PID" "http://127.0.0.1:$PORT/health" "Generator server"
  wait_for_server "$EMBED_PID" "http://127.0.0.1:$EMBED_PORT/v1/models" "Embedding server"
  echo "Both servers are online."

  wait "$GEN_PID" "$EMBED_PID"
fi
