#!/bin/bash

#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=64gb
#SBATCH --time=0-01:00:00
#SBATCH --job-name=embed_prompts
#SBATCH --output=Agents/out/embed_prompts.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

source ~/disk/venvs/pnw-2/bin/activate 

GET_CONFIG="python3 Agents/utils/config.py"

MY_DISK=$($GET_CONFIG paths.base_path)
PROMPTS_DIR=$MY_DISK/$($GET_CONFIG paths.prompts_dir) 

export TORCHINDUCTOR_CACHE_DIR=$MY_DISK/.cache/torch_inductor
export VLLM_CACHE_ROOT=$MY_DISK/.cache/vllm
export HF_HOME=$MY_DISK/.cache/hf
export TRITON_CACHE_DIR=$MY_DISK/.cache/triton
export FLASHINFER_CACHE_DIR=$MY_DISK/.cache/flashinfer
export TORCH_EXTENSIONS_DIR=$MY_DISK/.cache/torch_extensions
export XDG_CACHE_HOME=$MY_DISK/.cache

echo "Configuration Loaded:"
echo "  Agents Root Path: $MY_DISK"
echo "  VLLM Compile Cache: $VLLM_CACHE_ROOT"
echo "  Prompts Path: $PROMPTS_DIR"

rm -rf "$VLLM_CACHE_ROOT"
mkdir -p "$VLLM_CACHE_ROOT"
mkdir -p "$PROMPTS_DIR"

echo "====================================="
echo "Starting Prompts Embedding..."
echo "====================================="

python3 Agents/src/embed_prompts.py

echo "Done!"