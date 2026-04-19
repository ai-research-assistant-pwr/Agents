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

source /home/tymrom7227/disk/venvs/pnw-2/bin/activate
VENV_PYTHON="/home/tymrom7227/disk/venvs/pnw-2/bin/python"

MY_DISK="/home/tymrom7227/disk"
AGENTS_DIR="$MY_DISK/Agents"

export XDG_CACHE_HOME=$MY_DISK/.cache
export HOME=$MY_DISK
export HF_HOME=$MY_DISK/.cache/hf
export PYTHONPATH="$AGENTS_DIR:$PYTHONPATH"

echo "=========================================="
echo "INSTALLING PYTORCH (CUDA 12.4)"
echo "=========================================="

$VENV_PYTHON -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo "=========================================="
echo "INSTALLING CORE LIBRARIES"
echo "=========================================="

$VENV_PYTHON -m pip install \
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
  ray[default]==2.48.0

echo "=========================================="
echo "INSTALLING DEEPSPEED (SAFE MODE)"
echo "=========================================="

DS_BUILD_OPS=0 $VENV_PYTHON -m pip install deepspeed==0.18.0

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

$VENV_PYTHON -m pip install -e . --no-deps

echo "=========================================="
echo "TESTING INSTALLATION"
echo "=========================================="

$VENV_PYTHON - <<EOF
import torch
print("Torch CUDA:", torch.cuda.is_available())

import transformers
print("Transformers OK")

import accelerate
print("Accelerate OK")
EOF

echo "=========================================="
echo "SETUP COMPLETED SUCCESSFULLY"
echo "=========================================="