import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MY_DISK = "/home/tymrom7227/disk"
BASE_MODEL_PATH = os.path.join(MY_DISK, "models/Qwen/Qwen3-4B-Instruct-2507")

ADAPTER_PATH = os.path.join(MY_DISK, "models_output/run5/lora_shared_agent_Qwen3-4B-Instruct-2507")

MERGED_SAVE_PATH = os.path.join(MY_DISK, "models_output/run5/Qwen3-4B-SFT-Shared-Merged")

def main():
    print(f"=== Merging LoRA weights for Shared Agent ===")
    
    if not os.path.exists(ADAPTER_PATH):
        print(f"ERROR: Adapter not found at {ADAPTER_PATH}")
        return

    print("Loading base model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    print(f"Applying Shared SFT (LoRA) weights from: {os.path.basename(ADAPTER_PATH)}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Merging weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {MERGED_SAVE_PATH}...")
    model.save_pretrained(MERGED_SAVE_PATH)
    tokenizer.save_pretrained(MERGED_SAVE_PATH)

    print("Applying patch to tokenizer_config.json...")
    config_path = os.path.join(MERGED_SAVE_PATH, "tokenizer_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "extra_special_tokens" in data and isinstance(data["extra_special_tokens"], list):
                del data["extra_special_tokens"]
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print("   -> Success: Fixed tokenizer_config.json.")
            else:
                print("   -> Patch not needed (config is clean).")
        except Exception as e:
            print(f"   -> Error applying patch: {e}")

    print("\n=== SUCCESS: Shared model merged and saved. Ready for GRPO ===")

if __name__ == "__main__":
    main()