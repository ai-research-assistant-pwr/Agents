import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen3-4B"
# Ścieżka, gdzie Twój run_sft.py zrzucił adapter
adapter_path = "./models_output/sft/lora_shared_agent_Qwen3-4B" 
output_path = "./models_output/merged_sft_qwen_4B"

print("Ładowanie modelu bazowego...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, device_map="cpu"
)
print("Aplikowanie adaptera LoRA...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Łączenie wag (merge and unload)...")
model = model.merge_and_unload()

print(f"Zapisywanie zmergowanego modelu do {output_path}...")
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(output_path)
print("Gotowe!")