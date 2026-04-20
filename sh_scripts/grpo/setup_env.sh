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

# =================================================
# DIRECTORY SETUP
# =================================================
mkdir -p "$MY_DISK/.tmp" "$MY_DISK/.cache/hf" "$MY_DISK/Agents/out"

export TMPDIR="$MY_DISK/.tmp"
export XDG_CACHE_HOME="$MY_DISK/.cache"
export HOME="$MY_DISK"
export HF_HOME="$MY_DISK/.cache/hf"
export PIP_CACHE_DIR="$MY_DISK/.cache/pip"
export MARTI_DIR="$MARTI_DIR"

# =================================================
# VENV
# =================================================
echo "-> Deleting old venv and creating new one (Python 3.11)..."
rm -rf "$VENV_PATH"
python -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

pip install --upgrade pip setuptools wheel packaging ninja

# =================================================
# PYTORCH cu121
# =================================================
echo "-> Installing PyTorch 2.4.0 + cu121..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# =================================================
# CLONE REPOSITORIES
# =================================================
echo "-> Cloning repositories..."

cd "$MY_DISK"

if [ ! -d "OpenRLHF" ]; then
    git clone https://github.com/OpenRLHF/OpenRLHF.git
fi

if [ ! -d "MARTI" ]; then
    git clone https://github.com/TsinghuaC3I/MARTI.git
fi

# =================================================
# INSTALL OPENRLHF
# =================================================
echo "-> Installing OpenRLHF..."
cd "$MY_DISK/OpenRLHF"
pip install -e .

# =================================================
# INSTALL MARTI
# =================================================
echo "-> Installing MARTI..."
cd "$MY_DISK/MARTI"
pip install -e . --no-deps

# =================================================
# DEPENDENCIES
# =================================================
echo "-> Installing Ray 2.48.0..."
pip install "ray[default]==2.48.0"

echo "-> Installing Transformers 4.57.0..."
pip install "transformers==4.57.0"

echo "-> Installing DeepSpeed 0.18.0..."
DS_BUILD_OPS=0 pip install deepspeed==0.18.0

echo "-> Installing remaining MARTI requirements..."
pip install \
    accelerate \
    bitsandbytes \
    datasets \
    einops \
    "grpcio>=1.74.0" \
    isort \
    jsonlines \
    loralib \
    optimum \
    "optree>=0.13.0" \
    peft \
    "pynvml>=12.0.0" \
    tensorboard \
    torchdata \
    torchmetrics \
    tqdm \
    transformers_stream_generator \
    wandb \
    pyyaml pydantic python-dotenv openai pandas

# =================================================
# vLLM 0.8.5
# =================================================
echo "-> Installing vLLM 0.8.5.post1..."
pip install "vllm==0.8.5.post1"

# =================================================
# FLASH-ATTN
# =================================================
echo "-> Installing Flash-Attn (prebuilt binary)..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# =================================================
# MARTI
# =================================================
echo "-> Installing MARTI package (editable, no-deps)..."
cd "$MARTI_DIR"
pip install -e . --no-deps

echo "-> Injecting .pth file for bulletproof linking..."
echo "$MARTI_DIR" > "$VENV_PATH/lib/python3.11/site-packages/marti.pth"

# =================================================
# verification
# =================================================
echo "-> Final Verification..."
export PYTHONPATH="$MARTI_DIR:${PYTHONPATH:-}"

"$VENV_PATH/bin/python" << 'PYEOF'
import sys, os

errors = []

checks = {
    'torch':        lambda: __import__('torch'),
    'openrlhf':     lambda: __import__('openrlhf'),
    'vllm':         lambda: __import__('vllm'),
    'ray':          lambda: __import__('ray'),
    'transformers': lambda: __import__('transformers'),
    'deepspeed':    lambda: __import__('deepspeed'),
    'flash_attn':   lambda: __import__('flash_attn'),
}

for name, fn in checks.items():
    try:
        mod = fn()
        ver = getattr(mod, '__version__', 'unknown')
        extra = ''
        if name == 'torch':
            extra = f', CUDA: {mod.cuda.is_available()}'
        if name == 'openrlhf':
            extra = f', path: {mod.__file__}'
        print(f'[OK] {name} {ver}{extra}')
    except ImportError as e:
        errors.append(f'{name}: {e}')

if errors:
    print('\nIMPORT ERRORS:')
    for err in errors:
        print(f'  [FAIL] {err}')
    sys.exit(1)

print('\nSUCCESS: All imports successful. MARTI environment is set up correctly.')
PYEOF

echo "=========================================="
echo "SETUP COMPLETED SUCCESSFULLY"
echo "=========================================="