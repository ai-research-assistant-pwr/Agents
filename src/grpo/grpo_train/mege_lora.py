import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
adapter_path = os.path.join(BASE_DIR, "models_output", "sft", "lora_shared_agent_Qwen3-4B")
output_path = os.path.join(BASE_DIR, "models_output", "merged_sft_qwen_4B")

print("Ładowanie modelu bazowego...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    torch_dtype=torch.float16,
    device_map="auto",
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