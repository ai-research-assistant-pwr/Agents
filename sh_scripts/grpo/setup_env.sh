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
echo "CREATING NEW ENV: pnw-3"
echo "=========================================="

python -m venv $VENV_PATH
source $VENV_PATH/bin/activate

pip install --upgrade pip setuptools wheel

export XDG_CACHE_HOME=$MY_DISK/.cache
export HOME=$MY_DISK
export HF_HOME=$MY_DISK/.cache/hf
export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

echo "=========================================="
echo "INSTALLING PYTORCH (CUDA 12.1 - vLLM SAFE)"
echo "=========================================="

pip install torch==2.3.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

echo "=========================================="
echo "INSTALLING vLLM"
echo "=========================================="

pip install vllm==0.5.4

echo "=========================================="
echo "INSTALLING CORE LIBRARIES"
echo "=========================================="

pip install \
  accelerate \
  datasets \
  einops \
  jsonlines \
  loralib \
  optimum \
  packaging \
  peft \
  pynvml \
  tensorboard \
  torchmetrics \
  tqdm \
  transformers==4.57.0 \
  transformers_stream_generator \
  wandb \
  ray[default]==2.48.0 \
  pydantic \
  python-dotenv \
  langchain-core \
  langchain-google-genai \
  pyyaml \
  google-genai \
  openai \
  cerebras-cloud-sdk \
  numpy \
  pandas

echo "=========================================="
echo "INSTALLING DEEPSPEED (SAFE MODE)"
echo "=========================================="

DS_BUILD_OPS=0 pip install deepspeed==0.18.0

echo "=========================================="
echo "INSTALLING OpenRLHF"
echo "=========================================="

pip install openrlhf

echo "=========================================="
echo "CLONING MARTI"
echo "=========================================="

cd $MY_DISK

if [ ! -d "MARTI" ]; then
    git clone https://github.com/TsinghuaC3I/MARTI.git
else
    echo "Directory MARTI already exists, skipping clone."
fi

cd $MY_DISK/MARTI

echo "=========================================="
echo "INSTALLING MARTI (NO AUTO-DEPS)"
echo "=========================================="

pip install -e . --no-deps

echo "=========================================="
echo "TESTING INSTALLATION"
echo "=========================================="

python - <<EOF
import torch
print("Torch CUDA:", torch.cuda.is_available())

import vllm
print("vLLM OK")

import transformers
print("Transformers OK")

import openrlhf
print("OpenRLHF OK")
EOF

echo "=========================================="
echo "SETUP COMPLETED SUCCESSFULLY"
echo "=========================================="