#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32gb
#SBATCH --time=0-00:30:00
#SBATCH --job-name=dense_retrieval
#SBATCH --output=Agents/out/retrieval.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

source /home/tymrom7227/disk/venvs/pnw-2/bin/activate
VENV_PYTHON="/home/tymrom7227/disk/venvs/pnw-2/bin/python"

AGENTS_DIR="$HOME/disk/Agents"
export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

GET_CONFIG="$VENV_PYTHON $AGENTS_DIR/src/sft/utils/config.py"
MY_DISK=$($GET_CONFIG paths.base_path)

export HOME=$MY_DISK
export XDG_CACHE_HOME=$MY_DISK/.cache

RERANKER_PORT=$($GET_CONFIG models.reranker_port)
RERANKER_MODEL=$($GET_CONFIG models.reranker_model)

echo "====================================="
echo "Starting vLLM Reranker Server..."
echo "====================================="

CUDA_VISIBLE_DEVICES=0 vllm serve $RERANKER_MODEL \
    --served-model-name $RERANKER_MODEL \
    --runner pooling \
    --port $RERANKER_PORT \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096 \
    --dtype bfloat16 &

VLLM_PID=$!

echo "Waiting for reranker to load model (port $RERANKER_PORT)..."
while ! curl -s http://localhost:$RERANKER_PORT/health > /dev/null; do
    sleep 5
    echo "  Reranker still loading..."
done
echo "Reranker is ready!"

echo "====================================="
echo "Starting Dense Retrieval + Reranking..."
echo "====================================="

python3 $AGENTS_DIR/scripts/sft/run_retrieval.py
PYTHON_EXIT=$?

echo "Shutting down vLLM server..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

echo "Done!"
exit $PYTHON_EXIT