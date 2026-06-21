#!/usr/bin/env bash
# Stop all vLLM servers started by serve_vllm.sh.
# Reads PIDs from tmp/ files written at startup.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TMP_DIR="${PROJECT_ROOT}/tmp"

PID_FILES=(
    "${TMP_DIR}/vllm_embedding.pid"
    "${TMP_DIR}/vllm_chat.pid"
    "${TMP_DIR}/vllm_monitor.pid"
)

PIDS=()
for pid_file in "${PID_FILES[@]}"; do
    if [ -f "${pid_file}" ]; then
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            PIDS+=("${pid}")
            echo "Found running PID ${pid} (${pid_file##*/})"
        else
            echo "PID ${pid} from ${pid_file##*/} is already gone — skipping"
        fi
    else
        echo "No PID file: ${pid_file##*/}"
    fi
done

if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "No running servers found."
    rm -f "${PID_FILES[@]}" 2>/dev/null || true
    exit 0
fi

echo "Sending SIGTERM to ${#PIDS[@]} process(es)..."
for pid in "${PIDS[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true   # whole process group
    kill -TERM "${pid}"      2>/dev/null || true   # process itself
done

# Wait up to 20 s for voluntary exit
deadline=$(( SECONDS + 20 ))
while [ "${SECONDS}" -lt "${deadline}" ]; do
    alive=0
    for pid in "${PIDS[@]}"; do
        kill -0 "${pid}" 2>/dev/null && alive=1
    done
    [ "${alive}" -eq 0 ] && break
    sleep 1
done

# Force-kill anything still running
for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
        echo "Force-killing PID ${pid}..."
        kill -KILL -- "-${pid}" 2>/dev/null || true
        kill -KILL "${pid}"      2>/dev/null || true
    fi
done

wait 2>/dev/null || true

# Remove PID files
rm -f "${PID_FILES[@]}" 2>/dev/null || true

echo "Done."
