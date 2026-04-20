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
module load Python/3.11.5-GCCcore-13.2.0 || module load Python/3.11.3-GCCcore-12.3.0

MY_DISK="/home/tymrom7227/disk"
VENV_PATH="$MY_DISK/venvs/pnw-3"
MARTI_DIR="$MY_DISK/MARTI"

echo "-> Deleting old venv and creating new one (Python 3.11)..."
rm -rf $VENV_PATH
python -m venv $VENV_PATH
source $VENV_PATH/bin/activate

mkdir -p $MY_DISK/.tmp
export TMPDIR=$MY_DISK/.tmp
export XDG_CACHE_HOME=$MY_DISK/.cache
export HOME=$MY_DISK
export HF_HOME=$MY_DISK/.cache/hf

pip install --upgrade pip setuptools wheel packaging ninja

echo "-> Installing PyTorch (cu121 has best wheels for 3.11)..."
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "-> Installing vLLM and Ray..."
pip install vllm==0.6.3.post1
pip install ray[default]==2.35.0

echo "-> Installing MARTI requirements..."
pip install bitsandbytes isort optree torchdata transformers_stream_generator
pip install "transformers>=4.45.2,<4.47.0"
pip install "opentelemetry-sdk>=1.26.0,<1.27.0" "opentelemetry-api>=1.26.0,<1.27.0"
DS_BUILD_OPS=0 pip install deepspeed==0.18.0

echo "-> Installing Flash-Attn (Direct Binary)..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

echo "-> Installing utilities..."
pip install wandb pyyaml pydantic python-dotenv openai pandas accelerate datasets einops peft

echo "-> Installing MARTI as openrlhf package..."
cd $MY_DISK
if [ ! -d "MARTI" ]; then
    git clone https://github.com/TsinghuaC3I/MARTI.git
fi

cd $MARTI_DIR
pip install --no-build-isolation --no-deps -e .

echo "-> Final Verification..."
export PYTHONPATH="$MARTI_DIR:$PYTHONPATH"

$VENV_PATH/bin/python -c "
import sys
import os
# Siłowe dodanie ścieżki na początek
sys.path.insert(0, '$MARTI_DIR')
try:
    import torch
    import openrlhf
    import vllm
    print('CUDA:', torch.cuda.is_available())
    print('OpenRLHF/MARTI path:', openrlhf.__file__)
    print('vLLM OK')
    print('SUCCESS: Wszystko działa!')
except ImportError as e:
    print(f'BŁĄD: Nie znaleziono modułu: {e}')
    print('Zawartość folderu MARTI:')
    os.system('ls -F $MARTI_DIR')
    sys.exit(1)
"

echo "=========================================="
echo "SETUP COMPLETED SUCCESSFULLY"
echo "=========================================="