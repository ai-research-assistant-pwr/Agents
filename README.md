# Agents-emcom

Multi-agent scientific hypothesis generation system training a shared Qwen3-4B model as both Retriever and Generator, with emergent communication (EmCom) techniques applied during RL fine-tuning.

---

## Overview

Training proceeds in two stages:

1. **SFT** — A single Qwen3-4B + LoRA adapter is trained on both Retriever and Generator roles (role separation via system-prompt injection), bootstrapped from Gemini/GPT demonstrations.
2. **GRPO** — The SFT checkpoint is fine-tuned with a multi-component reward signal and two EmCom pressures:
   - **Information Bottleneck length penalty (λ)** — subtracts `λ · token_count` of the Retriever's message from the reward, incentivising compression.
   - **Stochastic channel noise** — randomly replaces word tokens with `[MASK]` before the Generator reads the message, pressuring the Retriever towards redundancy-tolerant outputs.

The repository also includes a modular inference pipeline backed by Weaviate (vector search), Neo4j (knowledge graph), and external LLM APIs.

---

## Architecture

### Sender–Receiver Environment

The multi-agent setup is a one-way, single-turn communication game. The Sender (Retriever) has full access to the hidden environment state (query + paper fragments) and must compress it into a message. The Receiver (Generator) sees only that message and produces scientific hypotheses that are scored against source knowledge.

![Sender-Receiver environment](diagram_sender_receiver.png)

### Shared Policy

A single Qwen3-4B + LoRA checkpoint plays both roles. Role assignment is done at inference time via system-prompt injection — no separate model copies are needed.

![Shared policy architecture](diagram_shared_policy.png)

### Channel Pressure (EmCom Techniques)

Two complementary pressures are applied to the communication channel during GRPO training to induce emergent compression:

![Channel pressure: bottleneck and noise](diagram_channel_pressure.png)

The reward signal is:

$$R = w_{\text{sim}} \cdot R_{\text{sim}} + w_{\text{div}} \cdot R_{\text{div}} + w_{\text{ground}} \cdot R_{\text{ground}} + w_{\text{rel}} \cdot R_{\text{rel}} - \lambda \cdot N_{\text{tokens}}$$

Embedding similarity is computed against a gold-label hypothesis using Qwen3-Embedding-4B; groundedness and relevancy use Qwen3-Reranker-0.6B. All weights and EmCom hyperparameters are configurable at launch time.

---

## Repository Structure

```
Agents-emcom/
├── config/
│   ├── app/config.yaml          # Inference pipeline config
│   ├── grpo/config.yaml         # GRPO hyperparameters
│   └── sft/config.yaml          # SFT hyperparameters
├── scripts/
│   ├── app/                     # Inference runners (single & batch)
│   ├── rl/                      # RL dataset construction
│   ├── sft/                     # SFT data generation & retrieval
│   └── hypotheses_evaluation/   # LLM-judge scoring scripts
├── src/
│   ├── app/                     # Inference pipeline (App orchestrator, explorers, API clients)
│   ├── eval/                    # Evaluation: noise probe, semantics, pragmatics, attention
│   ├── grpo/grpo_train/         # Workflow, rewards, transitions, turn execution
│   ├── hypotheses_evaluation/   # Clarity / groundedness / relevancy judges
│   └── sft/sft_train/           # SFT training, dataset preparation, evaluation
├── sh_scripts/
│   ├── grpo/                    # SLURM job scripts (training, evaluation)
│   └── sft/                     # SLURM job scripts (SFT)
└── pyproject.toml
```

---

## Installation

**Requirements:** Python 3.12, CUDA 12.8, Weaviate instance.

```bash
# PyTorch (CUDA 12.8)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# FlashAttention 2 (precompiled wheel)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# MARTI (GRPO training infrastructure)
git clone git@github.com:ai-research-assistant-pwr/marti-fork.git
cd marti-fork && pip install -r requirements.txt && pip install -e . && cd ..

pip install json5 srsly vllm==0.15.1
pip install -e .
```

For API-only inference, `pip install -e .` is sufficient. Set keys in `.env`:

```env
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
CEREBRAS_API_KEY=...
```

---

## Usage

**Inference**

```bash
python scripts/app/run_pipeline.py --prompt "What are the mechanisms of CRISPR off-target effects?"
python scripts/app/run_pipeline_batch.py --input queries.jsonl --output results/
```

**GRPO training** (SLURM, heterogeneous job: 4× H100 training + 1× H100 embedding/reranker)

```bash
# Default EmCom config
bash sh_scripts/grpo/run_grpo.sh $WANDB_API_KEY $WEAVIATE_NODE_ID

# Custom hyperparameters
LENGTH_PENALTY_LAMBDA=0.0005 NOISE_PROBABILITY=0.1 \
    bash sh_scripts/grpo/run_grpo.sh $WANDB_API_KEY $WEAVIATE_NODE_ID
```

**Evaluation**

```bash
# Generate clean and noisy trajectory sets
bash sh_scripts/grpo/run_eval_clean.sh && bash sh_scripts/grpo/run_eval_noisy.sh

# Statistical noise probe (H1: reward, H2: cosine sim to gold, H3: hypothesis divergence)
python src/eval/eval_noise_probe.py \
    --dir_a eval_clean/ --dir_b eval_noisy/ --output_dir eval_results/

# Semantic structure over training (signal dispersion, grounding index)
python src/eval/eval_semantics_v2.py --logs_dir workflow_logs/ --output_dir eval_results/semantics/
```

---

## Tech Stack

| | |
|---|---|
| **Model** | Qwen3-4B-Instruct, Qwen3-Embedding-4B, Qwen3-Reranker-0.6B |
| **Training** | PyTorch, HuggingFace Transformers + TRL, PEFT / LoRA, MARTI (GRPO) |
| **Inference** | vLLM 0.15.1, FlashAttention 2 |
| **Knowledge layer** | Weaviate (vector search), Neo4j (BFS / PageRank / Random Walk / Agentic) |
| **External APIs** | Google Gemini, OpenAI, Cerebras |
| **Evaluation** | SciPy (Mann-Whitney U, Wilcoxon), Matplotlib, Seaborn |
| **Infrastructure** | SLURM / WCSS HPC, CUDA 12.8, Weights & Biases |

---

## License

Released for academic and research purposes. See [LICENSE](LICENSE).

*Master's thesis project — Wrocław University of Science and Technology.*