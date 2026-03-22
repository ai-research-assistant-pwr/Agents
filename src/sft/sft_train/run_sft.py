import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.dirname(SFT_DIR)
AGENTS_DIR = os.path.dirname(SRC_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.sft.utils.config import CONFIG

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT Training")
    parser.add_argument("--task", type=str, required=True, choices=["retriever", "generator"], 
                        help="Which agent to train: 'retriever' or 'generator'")
    return parser.parse_args()

def main():
    args = parse_args()
    task = args.task
    
    print(f"\n{'='*50}")
    print(f" Running SFT Training for: {task.upper()}")
    print(f"{'='*50}\n")

    DATASETS_DIR = os.path.join(AGENTS_DIR, "data", "datasets")
    TRAIN_FILE = os.path.join(DATASETS_DIR, f"{task}_train.jsonl")
    EVAL_FILE = os.path.join(DATASETS_DIR, f"{task}_eval.jsonl")
    
    base_model_key = f"{task}_base_model"
    
    if base_model_key not in CONFIG["training"]:
        raise ValueError(f"ERROR: No key '{base_model_key}' found in config under 'training'. Please specify the base model for {task} in the config.yaml.")
        
    BASE_MODEL_ID = CONFIG["training"][base_model_key]
    MODELS_DIR = os.path.join(CONFIG["paths"]["base_path"], CONFIG["paths"]["models_dir"])
    MODEL_PATH = os.path.join(MODELS_DIR, BASE_MODEL_ID)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Didn't find local model at {MODEL_PATH}. Will attempt to load from Hugging Face Hub: {BASE_MODEL_ID}")
        MODEL_PATH = BASE_MODEL_ID
    else:
        print(f"✅ Found local base model: {MODEL_PATH}")
       
    safe_model_name = BASE_MODEL_ID.split("/")[-1]
    OUTPUT_DIR = os.path.join(CONFIG["paths"]["base_path"], CONFIG["paths"]["models_output_dir"], f"lora_{task}_{safe_model_name}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # W&B CONFIG
    run_name = f"{task}-sft-{safe_model_name}-lr{CONFIG['training']['learning_rate']}"
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank <= 0:
        wandb.init(
            project=CONFIG["training"].get("wandb_project", "agents_sft_training"),
            name=run_name,
            tags=["sft", task, "emergent-comm"],
            config=CONFIG,
            reinit=True
        )

    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"

    print(f"Loading data from {DATASETS_DIR}...")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "test": EVAL_FILE})

    # Loading Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def formatting_prompts_func(example):
        return [tokenizer.apply_chat_template(msg, tokenize=False) for msg in example['messages']]

    # Loss masking
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    print("Loading model to VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False 

    peft_config = LoraConfig(
        r=CONFIG["training"].get("lora_r", 16),
        lora_alpha=CONFIG["training"].get("lora_alpha", 32),
        lora_dropout=0.05,
        target_modules=CONFIG["training"].get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=CONFIG["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["training"]["gradient_accumulation_steps"],
        learning_rate=float(CONFIG["training"]["learning_rate"]),
        num_train_epochs=CONFIG["training"]["num_train_epochs"],
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        report_to="wandb",
        run_name=run_name,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=CONFIG["training"]["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    wandb.finish()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()