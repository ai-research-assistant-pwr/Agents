import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = "Agents"
BASE_MODEL_NAME = "Qwen/Qwen3-4B"

# Ścieżki do modeli
ADAPTER_PATH = os.path.join(BASE_DIR, "models_output", "sft", "lora_shared_agent_Qwen3-4B")
OUTPUT_PATH = os.path.join(BASE_DIR, "models_output", "merged_sft_qwen_4B")

def merge_lora_weights():
    print(f"Checking for adapter at: {ADAPTER_PATH}")
    if not os.path.exists(ADAPTER_PATH):
        print(f"ERROR: Adapter path not found: {ADAPTER_PATH}")
        sys.exit(1)

    print(f"=> Loading base model: {BASE_MODEL_NAME}...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        
        print("=> Applying LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

        print("=> Merging weights (merge_and_unload)...")
        model = model.merge_and_unload()

        print(f"=> Saving merged model to: {OUTPUT_PATH}...")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        model.save_pretrained(OUTPUT_PATH)
        
        print("=> Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        tokenizer.save_pretrained(OUTPUT_PATH)

        print("\n=> SUCCESS: Model successfully merged.")
        print(f"   Saved at: {OUTPUT_PATH}")

    except Exception as e:
        print(f"\n=> ERROR during merging: {e}")
        sys.exit(1)

if __name__ == "__main__":
    merge_lora_weights()