import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ścieżki
base_model_path = "/home/tymrom7227/disk/models/Qwen/Qwen3-4B-Instruct-2507"
adapter_path = "/home/tymrom7227/disk/models_output/run4/lora_generator_Qwen3-4B-Instruct-2507"
merged_save_path = "/home/tymrom7227/disk/models_output/run4/Qwen3-4B-SFT-Merged"

print("Loading base model and tokenizer...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print("Applying SFT (LoRA) weights...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging weights into base model...")
model = model.merge_and_unload()

print(f"Saving to directory: {merged_save_path}...")
model.save_pretrained(merged_save_path)
tokenizer.save_pretrained(merged_save_path)

print("Applying patch to tokenizer_config.json...")
config_path = os.path.join(merged_save_path, "tokenizer_config.json")
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
            print("   -> Patch not needed (everything is fine).")
    except Exception as e:
        print(f"   -> Error applying patch: {e}")

print("Success Weights merged and model saved.")