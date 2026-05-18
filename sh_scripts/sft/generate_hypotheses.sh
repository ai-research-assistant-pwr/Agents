#!/bin/bash

#SBATCH -N 1    # CPU nodes
#SBATCH -c 4    # number of CPU cores
#SBATCH --mem=64gb
#SBATCH --time=0-04:00:00    # format dd-hh:mm:ss np. 1-12:30:15
#SBATCH --job-name=gen_hypotheses
#SBATCH --output=out/gen_hypotheses.out
#SBATCH -p lem-gpu-normal    # partition
#SBATCH --gres=gpu:hopper:2 

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
source ~/disk/patryk/Agents/qwen_venv/bin/activate

export MY_DISK=/home/$USER/disk
export MY_NEW_TMP="$MY_DISK/patryk/tmp"
export XDG_CACHE_HOME="$MY_NEW_TMP/xdg_cache"
export TRITON_CACHE_DIR="$MY_NEW_TMP/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$MY_NEW_TMP/torchinductor_cache"

vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 11000 \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 &

VLLM_PID=$!

echo "Waiting for vLLM to load model..."
while ! curl -s http://localhost:8000/health > /dev/null; do
    sleep 10
    echo "Still loading..."
done
echo "vLLM is ready!"

echo "Running generator inference..."
python scripts/rl/run_generator_inference.py

kill $VLLM_PID
