#!/usr/bin/env bash
# Start vLLM for the locally fine-tuned Qwen3-4B model.
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

echo "Model weights : $WEIGHTS"
echo "Port          : $PORT"
echo ""

if $USE_DOCKER; then
  # Convert the host-relative weights path (models/…) to the container path (/models/…).
  # The docker-compose volume mounts {project_root}/models as /models inside the container.
  CONTAINER_MODEL="/${WEIGHTS%/}"   # "models/foo/bar" → "/models/foo/bar"
  export VLLM_MODEL="$CONTAINER_MODEL"
  export VLLM_PORT="$PORT"
  echo "Starting vLLM via Docker Compose (VLLM_MODEL=$VLLM_MODEL, VLLM_PORT=$VLLM_PORT)"
  docker compose -f "$PROJECT_ROOT/docker/vllm/docker-compose.yml" up
else
  #HOST_MODEL="$PROJECT_ROOT/$WEIGHTS"
  HOST_MODEL=$WEIGHTS
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
      --gpu-memory-utilization 0.85 \
      --max-model-len 8192 \
      --max-num-seqs 4 \
      --enforce-eager \
      --reasoning-parser qwen3 \
      --port "$PORT" \
      --host 0.0.0.0
fi
