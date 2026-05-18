"""
run_sft.py
==========
SFT training for the unified Retriever + Generator LoRA.

Key design decisions
--------------------
1.  Qwen3 thinking format
    The assistant turn is always:
        <think>\\n{reasoning}\\n</think>\\n\\n{output}
    We keep the model's default chat_template (Qwen3 already handles <think>).
    We do NOT override the template — overriding it broke think-token handling
    in earlier experiments.

2.  Unified LoRA
    One adapter trains on both retriever and generator records (interleaved by
    prepare_dataset.py).  After merging with the base model this becomes the
    starting checkpoint for GRPO.

3.  assistant_only_loss = True
    Gradients flow only through the assistant turn (system + user are masked).
    The reasoning inside <think>...</think> IS part of the assistant turn and
    therefore DOES receive gradient — the model learns to reason.

4.  Eval split
    If shared_agent_eval.jsonl exists next to the train file it is used for
    eval_loss tracking.  If it does not exist, evaluation is disabled and
    save_strategy switches to epoch-based to still save checkpoints.

Usage
-----
    python run_sft.py [--task shared_agent]
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_DIR    = os.path.dirname(SCRIPT_DIR)
SRC_DIR    = os.path.dirname(SFT_DIR)
AGENTS_DIR = os.path.dirname(SRC_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.sft.utils.config import CONFIG

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="SFT — Unified Retriever+Generator LoRA")
    parser.add_argument(
        "--task", type=str, default="shared_agent",
        help="Dataset prefix (default: 'shared_agent').  Files expected: "
             "<prefix>_train.jsonl and optionally <prefix>_eval.jsonl"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# chat_template validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_template(tokenizer) -> None:
    """
    Smoke-test the tokenizer's chat_template with a minimal thinking-format
    assistant turn.  Prints a warning if <think> tokens are not present in the
    rendered output (means the template does not support thinking format).
    """
    sample = [
        {"role": "system",    "content": "You are a test agent."},
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "<think>\nsome reasoning\n</think>\n\nsome output"},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            sample, tokenize=False, add_generation_prompt=False
        )
        if "<think>" not in rendered:
            print(
                "[WARN] The tokenizer's chat_template does not preserve <think> tags. "
                "Verify that you are using a Qwen3 tokenizer that supports thinking format."
            )
        else:
            print("[OK] chat_template handles <think> tokens correctly.")
    except Exception as e:
        print(f"[WARN] chat_template validation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    task = args.task

    print(f"\n{'='*55}")
    print(f" SFT Training — task: {task.upper()}")
    print(f"{'='*55}\n")

    # ── paths ─────────────────────────────────────────────────────────────────
    DATASETS_DIR = os.path.join(AGENTS_DIR, CONFIG["paths"]["dataset_prepped"])
    TRAIN_FILE   = os.path.join(DATASETS_DIR, f"{task}_train.jsonl")
    EVAL_FILE    = os.path.join(DATASETS_DIR, f"{task}_eval.jsonl")

    has_eval = os.path.exists(EVAL_FILE)
    if not has_eval:
        print(f"[INFO] Eval file not found at {EVAL_FILE} — training without evaluation.")

    if not os.path.exists(TRAIN_FILE):
        print(f"ERROR: Train file not found: {TRAIN_FILE}")
        print("Run prepare_dataset.py first.")
        sys.exit(1)

    # ── base model ────────────────────────────────────────────────────────────
    base_model_key = f"{task}_base_model"
    if base_model_key in CONFIG["training"]:
        BASE_MODEL_ID = CONFIG["training"][base_model_key]
    elif "generator_base_model" in CONFIG["training"]:
        BASE_MODEL_ID = CONFIG["training"]["generator_base_model"]
        print(f"[INFO] '{base_model_key}' not in config — using 'generator_base_model'.")
    else:
        raise ValueError("No base model key found in config.yaml under 'training'.")

    MODELS_DIR = os.path.join(CONFIG["paths"]["base_path"], CONFIG["paths"]["models_dir"])
    MODEL_PATH = os.path.join(MODELS_DIR, BASE_MODEL_ID)
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Local model not found at {MODEL_PATH} — loading from HF Hub.")
        MODEL_PATH = BASE_MODEL_ID
    else:
        print(f"[OK] Local base model: {MODEL_PATH}")

    safe_model_name = BASE_MODEL_ID.split("/")[-1]
    OUTPUT_DIR = os.path.join(
        CONFIG["paths"]["base_path"],
        CONFIG["paths"]["models_output_dir"],
        f"lora_{task}_{safe_model_name}",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── W&B ───────────────────────────────────────────────────────────────────
    WANDB_LOGS_DIR = os.path.join(AGENTS_DIR, "wandb")
    os.makedirs(WANDB_LOGS_DIR, exist_ok=True)

    run_name   = f"{task}-sft-{safe_model_name}-lr{CONFIG['training']['learning_rate']}"
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"]     = "false"

    # ── dataset ───────────────────────────────────────────────────────────────
    print(f"Loading data from {DATASETS_DIR} ...")
    data_files = {"train": TRAIN_FILE}
    if has_eval:
        data_files["eval"] = EVAL_FILE

    dataset = load_dataset("json", data_files=data_files)
    print(f"  Train: {len(dataset['train'])} records")
    if has_eval:
        print(f"  Eval:  {len(dataset['eval'])} records")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # !! DO NOT override tokenizer.chat_template for Qwen3 !!
    # Qwen3's built-in template already handles <think>...</think> correctly.
    _validate_template(tokenizer)

    # ── model ─────────────────────────────────────────────────────────────────
    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r           = CONFIG["training"].get("lora_r",     16),
        lora_alpha  = CONFIG["training"].get("lora_alpha", 32),
        lora_dropout= 0.05,
        target_modules = CONFIG["training"].get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        bias      = "none",
        task_type = "CAUSAL_LM",
    )

    # ── training args ─────────────────────────────────────────────────────────
    # Adapt save/eval strategy depending on whether we have an eval set
    if has_eval:
        eval_strategy  = "steps"
        save_strategy  = "steps"
        eval_steps     = CONFIG["training"].get("eval_steps", 50)
        save_steps     = CONFIG["training"].get("save_steps", 50)
        load_best      = True
        metric_best    = "eval_loss"
        greater_better = False
    else:
        eval_strategy  = "no"
        save_strategy  = "epoch"
        eval_steps     = None
        save_steps     = None
        load_best      = False
        metric_best    = None
        greater_better = False

    sft_config_kwargs = dict(
        output_dir                      = OUTPUT_DIR,
        per_device_train_batch_size     = CONFIG["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps     = CONFIG["training"]["gradient_accumulation_steps"],
        learning_rate                   = float(CONFIG["training"]["learning_rate"]),
        num_train_epochs                = CONFIG["training"]["num_train_epochs"],
        logging_steps                   = 1,
        eval_strategy                   = eval_strategy,
        save_strategy                   = save_strategy,
        load_best_model_at_end          = load_best,
        metric_for_best_model          = metric_best,
        greater_is_better               = greater_better,
        bf16                            = True,
        optim                           = "adamw_torch",
        report_to                       = "wandb",
        run_name                        = run_name,
        lr_scheduler_type               = "cosine",
        warmup_ratio                    = 0.1,
        max_length                      = CONFIG["training"]["max_seq_length"],
        # assistant_only_loss masks system+user tokens — only assistant tokens
        # (including <think> reasoning) contribute to the training loss.
        assistant_only_loss             = True,
        gradient_checkpointing          = True,
        gradient_checkpointing_kwargs   = {"use_reentrant": False},
    )

    if has_eval:
        sft_config_kwargs["eval_steps"] = eval_steps
        sft_config_kwargs["save_steps"] = save_steps

    training_args = SFTConfig(**sft_config_kwargs)

    # ── W&B init ──────────────────────────────────────────────────────────────
    if local_rank <= 0:
        wandb.init(
            project = CONFIG["training"].get("wandb_project", "agents_sft_training"),
            name    = run_name,
            dir     = WANDB_LOGS_DIR,
            tags    = ["sft", task, "unified_lora", "qwen3_thinking"],
            config  = CONFIG,
            reinit  = True,
        )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer_kwargs = dict(
        model            = model,
        train_dataset    = dataset["train"],
        peft_config      = peft_config,
        args             = training_args,
        processing_class = tokenizer,
    )
    if has_eval:
        trainer_kwargs["eval_dataset"] = dataset["eval"]

    trainer = SFTTrainer(**trainer_kwargs)

    # ── train ─────────────────────────────────────────────────────────────────
    print("\nStarting training ...")
    trainer.train()

    print(f"\nSaving LoRA adapter to {OUTPUT_DIR} ...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)   # save tokenizer alongside adapter

    if local_rank <= 0:
        wandb.finish()

    print("Training completed successfully.")
    print(f"Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()