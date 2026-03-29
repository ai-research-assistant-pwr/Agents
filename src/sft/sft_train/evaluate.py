import os
import sys
import json
import re
import argparse
from datetime import datetime
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_DIR    = os.path.dirname(SCRIPT_DIR)
SRC_DIR    = os.path.dirname(SFT_DIR)
AGENTS_DIR = os.path.dirname(SRC_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.sft.utils.config import CONFIG

DATASETS_DIR = os.path.join(AGENTS_DIR, CONFIG["paths"]["dataset_prepped"])
MODELS_DIR = os.path.join(CONFIG["paths"]["base_path"], CONFIG["paths"]["models_dir"])
MODELS_OUT_DIR = os.path.join(CONFIG["paths"]["base_path"], CONFIG["paths"]["models_output_dir"])
EVAL_OUT_DIR = os.path.join(AGENTS_DIR, "data", "eval_results")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

def extract_tag(text: str, tag: str) -> str:
    """Extracts content between <tag> and </tag> from the given text. Returns empty string if not found."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def check_format(text: str, required_tags: list) -> bool:
    """Checks if all required tags are present in the generated text."""
    for tag in required_tags:
        if f"<{tag}>" not in text or f"</{tag}>" not in text:
            return False
    return True

def calculate_grounding(hypothesis: str, extracted_info_str: str) -> float:
    """Checks how many of the variables from the extracted info are mentioned in the hypothesis. Returns a score between 0 and 1."""
    if not hypothesis or not extracted_info_str:
        return None
    try:
        info = json.loads(extracted_info_str)
        variables = info.get("variables", [])
    except json.JSONDecodeError:
        return None

    if not variables:
        return 1.0

    hyp_lower = hypothesis.lower()
    matched = sum(1 for v in variables if any(word.lower() in hyp_lower for word in v.split()))
    return matched / len(variables)

def extract_json_from_user_prompt(user_content: str, marker: str) -> str:
    """Extracts a JSON block with variables from the user prompt."""
    idx = user_content.find(marker)
    if idx == -1:
        return ""
    after = user_content[idx + len(marker):]
    next_marker = re.search(r"\n[A-Z][A-Z ]+:", after)
    if next_marker:
        after = after[:next_marker.start()]
    return after.strip()

def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight SFT Evaluation")
    parser.add_argument("--task", type=str, required=True, choices=["retriever", "generator"], 
                        help="Which agent to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["eval", "test"],
                        help="Dataset split (default: test)")
    return parser.parse_args()

def main():
    args = parse_args()
    task = args.task
    split = args.split
    
    print(f"\n{'='*50}")
    print(f" FAST EVALUATION: {task.upper()} on {split.upper()} split")
    print(f"{'='*50}\n")

    TEST_FILE = os.path.join(DATASETS_DIR, f"{task}_{split}.jsonl")
    if not os.path.exists(TEST_FILE):
        print(f"ERROR: File not found: {TEST_FILE}")
        sys.exit(1)

    base_model_key = f"{task}_base_model"
    BASE_MODEL_ID = CONFIG["training"][base_model_key]
    safe_name = BASE_MODEL_ID.split("/")[-1]
    
    LOCAL_BASE_PATH = os.path.join(MODELS_DIR, BASE_MODEL_ID)
    if not os.path.exists(LOCAL_BASE_PATH):
        LOCAL_BASE_PATH = BASE_MODEL_ID
        
    LORA_PATH = os.path.join(MODELS_OUT_DIR, f"lora_{task}_{safe_name}")
    if not os.path.exists(LORA_PATH):
        print(f"ERROR: LoRA adapter not found at {LORA_PATH}. Run SFT first!")
        sys.exit(1)

    records = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} test records.")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_BASE_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_BASE_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    if task == "retriever":
        flag_tag = "is_sufficient"
        required_tags = ["is_sufficient", "reasoning", "extracted_information"]
    else:
        flag_tag = "is_answerable"
        required_tags = ["is_answerable", "reasoning", "hypothesis", "natural_hypothesis", "falsification_criteria"]

    results_log = []
    correct_flags = 0
    correct_formats = 0
    grounding_scores = []

    print("\nStarting inference...")
    for rec in tqdm(records):
        prompt_id = rec.get("prompt_id", "unknown")
        messages = rec["messages"]
        
        input_msgs = [m for m in messages if m["role"] != "assistant"]
        gt_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        
        extracted_info_json = ""
        if task == "generator":
            user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
            extracted_info_json = extract_json_from_user_prompt(user_content, "EXTRACTED:")

        prompt = tokenizer.apply_chat_template(input_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        pred_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        gt_flag = extract_tag(gt_msg, flag_tag)
        pred_flag = extract_tag(pred_text, flag_tag)
        
        is_flag_correct = (gt_flag.lower() == pred_flag.lower()) if gt_flag and pred_flag else False
        is_format_correct = check_format(pred_text, required_tags)
        
        if is_flag_correct: correct_flags += 1
        if is_format_correct: correct_formats += 1

        grounding_val = None
        if task == "generator" and pred_flag.lower() == "true":
            pred_hyp = extract_tag(pred_text, "hypothesis")
            grounding_val = calculate_grounding(pred_hyp, extracted_info_json)
            if grounding_val is not None:
                grounding_scores.append(grounding_val)

        results_log.append({
            "prompt_id": prompt_id,
            "gt_flag": gt_flag,
            "pred_flag": pred_flag,
            "flag_correct": is_flag_correct,
            "format_correct": is_format_correct,
            "grounding_rate": grounding_val,
            "generated_text": pred_text
        })

    acc = (correct_flags / len(records)) * 100
    fmt = (correct_formats / len(records)) * 100
    avg_grounding = (sum(grounding_scores) / len(grounding_scores) * 100) if grounding_scores else 0.0

    print(f"\n{'='*50}")
    print(f" QUICK EVALUATION RESULTS: {task.upper()}")
    print(f"{'='*50}")
    print(f"Total Test Cases:   {len(records)}")
    print(f"Format Compliance:  {fmt:.2f}% (All XML tags successfully generated)")
    print(f"Boolean Accuracy:   {acc:.2f}% (Correctly predicted <{flag_tag}>)")
    if task == "generator":
        print(f"Avg Grounding Rate: {avg_grounding:.2f}% (Variables used in hypothesis)")
    print(f"{'='*50}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(EVAL_OUT_DIR, f"fast_eval_{task}_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": {"accuracy": acc, "format": fmt, "grounding": avg_grounding}, "details": results_log}, f, indent=4, ensure_ascii=False)
        
    print(f"Detailed logs saved to: {output_file}")

if __name__ == "__main__":
    main()