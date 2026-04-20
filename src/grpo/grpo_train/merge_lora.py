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

print("Applying SFT (LoRA) adapter weights...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging LoRA weights into the base model...")
model = model.merge_and_unload()

print(f"Saving to directory: {merged_save_path}...")
model.save_pretrained(merged_save_path)
tokenizer.save_pretrained(merged_save_path)

print("Merging complete. Model and tokenizer saved to:", merged_save_path)