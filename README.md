# Agents

...

# RL
## Venv setup on WCSS

Load the necessary modules. We want to use cuda 12.8 as it is recommended for flash attention 2.8.3.

    module load CUDA/12.8.0
    module load Python/3.12.3-GCCcore-13.3.0

Create a virtual environment.

    python -m venv venv
    source venv/bin/activate

Install torch 2.9.1 for cuda 12.8, as it is latest torch version, for which flash attention 2.8.3 has precompiled wheel.

    pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

Download precompiled flash attention 2.8.3 wheel for torch 2.9 and cuda 12.8, and install it.

    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

Clone and install MARTI.

    git clone git@github.com:ai-research-assistant-pwr/marti-fork.git
    cd marti-fork
    pip install -r requirements.txt
    pip install -e .

At last, install some missing dependencies.

    pip install json5 srsly vllm==0.15.1


## Run training

Use script in `sh_scripts/grpo/run_grpo.sh` to run training. You can modify the script to change hyperparameters, or set in command line, e.g.

    SIMILARITY_WEIGHT=0.5 DIVERSITY_WEIGHT=0.5 bash sh_scripts/grpo/run_grpo.sh

Script starts two nodes, one for training and one for helper models, that are used for evaluation and tools use.
