import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))

adapter_path = os.path.join(ROOT_DIR, "models_output", "sft", "lora_shared_agent_Qwen3-4B")
output_path = os.path.join(ROOT_DIR, "models_output", "merged_sft_qwen_4B")

print("Ładowanie modelu bazowego na CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)
print("Aplikowanie adaptera LoRA...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Łączenie wag (merge and unload)...")
model = model.merge_and_unload()

print(f"Zapisywanie zmergowanego modelu do {output_path}...")
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
tokenizer.save_pretrained(output_path)
print("Gotowe!")