#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=0-01:30:00
#SBATCH --job-name=setup_marti
#SBATCH --output=/home/tymrom7227/disk/Agents/out/setup_marti.out
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:1

set -e

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

MY_DISK="/home/tymrom7227/disk"
VENV_PATH="$MY_DISK/venvs/pnw-3"
AGENTS_DIR="$MY_DISK/Agents"

echo "=========================================="
echo "CLEANING AND UPDATING ENV: pnw-3"
echo "=========================================="

if [ ! -d "$VENV_PATH" ]; then
    python -m venv $VENV_PATH
fi

mkdir -p $MY_DISK/.tmp
export TMPDIR=$MY_DISK/.tmp

source $VENV_PATH/bin/activate

pip uninstall openrlhf -y || true

pip install --upgrade pip setuptools wheel packaging ninja

export XDG_CACHE_HOME=$MY_DISK/.cache
export HOME=$MY_DISK
export HF_HOME=$MY_DISK/.cache/hf
export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

echo "-> Installing PyTorch 2.4.0 (Better for Python 3.12 and Hopper)"
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "-> Installing core dependencies (fixed versions for stability)"
pip install vllm==0.6.3.post1
pip install ray==2.31.0
pip install "opentelemetry-sdk>=1.26.0,<1.27.0" "opentelemetry-api>=1.26.0,<1.27.0"

echo "-> Installing other libraries"
pip install \
  accelerate datasets einops jsonlines loralib optimum \
  peft pynvml tensorboard torchmetrics tqdm transformers==4.44.2 \
  wandb pyyaml pydantic python-dotenv openai pandas flash-attn --no-build-isolation

echo "-> Managing MARTI repository"
cd $MY_DISK
if [ ! -d "MARTI" ]; then
    git clone https://github.com/TsinghuaC3I/MARTI.git
fi

cd $MY_DISK/MARTI
pip install -e . --no-deps

echo "-> Final Verification"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); import openrlhf; print('OpenRLHF/MARTI OK'); import vllm; print('vLLM OK')"

echo "SETUP COMPLETED SUCCESSFULLY"