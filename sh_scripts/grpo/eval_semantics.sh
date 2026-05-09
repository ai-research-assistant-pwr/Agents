#!/bin/bash
# =================================================
# Evaluation Job: Experiment 2 (Semantics / TopSim)
# =================================================
#SBATCH --job-name=eval_exp2
#SBATCH --output=Agents/out/eval_semantics.out
#SBATCH --time=0-01:30:00
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:hopper:1

set -e 

MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"
VENV_PATH="$BASE_DIR/venv"

export EMBED_MODEL="Qwen/Qwen3-Embedding-4B"
export EMBED_PORT=8001
export EMBED_HOST="localhost"

source /usr/local/sbin/modules.sh
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0

source $VENV_PATH/bin/activate
VENV_PYTHON="$VENV_PATH/bin/python"

echo "=> Installing missing dependencies (Levenshtein)..."
$VENV_PYTHON -m pip install python-Levenshtein requests pandas matplotlib numpy scipy

# =================================================
# Setting up vLLM for TopSim Metric
# =================================================
echo "=> Setting up vLLM for TopSim Metric..."
$VENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL" \
    --host 0.0.0.0 \
    --port $EMBED_PORT \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.5 &
VLLM_PID=$!

echo "=> Waiting for vLLM..."
while ! curl -s http://$EMBED_HOST:$EMBED_PORT/v1/models > /dev/null; do
    sleep 5
done
echo "=> vLLM ready"

# =================================================
# 2. Uruchomienie Ewaluacji
# =================================================
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

echo "=> Starting TopSim calculations..."
$VENV_PYTHON "$BASE_DIR/src/eval/eval_semantics.py"

echo "=> Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null

echo "=> Evaluation of Experiment 2 completed successfully. Check Agents/eval_results/experiment_2."