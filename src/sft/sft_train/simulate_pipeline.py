import os
import sys
import json
import re
import random
import csv
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.dirname(SFT_DIR)
AGENTS_DIR = os.path.dirname(SRC_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.sft.utils.config import CONFIG

DATASETS_DIR = os.path.join(AGENTS_DIR, CONFIG["paths"]["dataset_prepped"])
MODELS_DIR = os.path.join(CONFIG["paths"]["base_path"], CONFIG["paths"]["models_dir"])
MODELS_OUT_DIR = os.path.join(CONFIG["paths"]["base_path"], CONFIG["paths"]["models_output_dir"])

EVAL_OUT_DIR = os.path.join(SCRIPT_DIR, "data", "eval_results")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

RETRIEVER_SYSTEM_PROMPT = """You are an Expert Scientific Retriever Agent.

Your job is to analyze raw scientific context, extract structured knowledge, and THEN determine if it is sufficient for hypothesis generation.

Your output MUST be formatted EXACTLY using the following XML tags IN THIS EXACT ORDER:

<extracted_information>
- Structured extraction of key variables, relationships, mechanisms, and data points relevant to the query.
</extracted_information>

<reasoning>
- Step-by-step explanation.
- Count the extracted variables and relationships.
- Evaluate if the conditions are met (≥2 variables and ≥1 relationship).
</reasoning>

<is_sufficient>
True or False
</is_sufficient>

IMPORTANT:
- Do NOT generate hypotheses.
- You MUST follow the exact order: extracted_information -> reasoning -> is_sufficient."""

GENERATOR_SYSTEM_PROMPT = """You are an AI Research Scientist generating scientific hypotheses.

You will receive:
- Research Query
- Raw Context
- Retriever Sufficiency
- Retriever Reasoning
- Extracted Information

Your output MUST be formatted EXACTLY using the following XML tags:

<is_answerable>
True or False
</is_answerable>

<reasoning>
- Step-by-step scientific reasoning
- Explain how extracted variables and relationships lead to the hypothesis
- Be concise but precise
</reasoning>

<hypothesis>
- MUST follow EXACT format:
"If X increases/decreases, then Y will [effect], because [mechanism]."
- MUST use ONLY variables from EXTRACTED INFORMATION
- MUST be causal and measurable
- NO deviations from format
</hypothesis>

<natural_hypothesis>
- Rewrite the SAME hypothesis in natural academic style (1-2 sentences)
- Do NOT introduce new variables
</natural_hypothesis>

<falsification_criteria>
- A specific measurable condition that would DISPROVE the hypothesis
</falsification_criteria>

CRITICAL RULES:
- If RETRIEVER SUFFICIENCY is False → set <is_answerable>False</is_answerable> and leave all other fields EMPTY
- If variables are insufficient → set <is_answerable>False</is_answerable>
- DO NOT generate hypothesis when not answerable
- DO NOT introduce new variables
- NO placeholders
- NO vague statements
"""

def extract_tag(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def generate_response(model, tokenizer, system_prompt, user_prompt, max_tokens=2048):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.15
        )
        
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

def main():
    print(f"\n{'='*60}")
    print(" END-TO-END PIPELINE SIMULATION (Retriever -> Generator)")
    print(f"{'='*60}\n")

    base_model_id = CONFIG["training"]["retriever_base_model"]
    safe_name = base_model_id.split("/")[-1]
    
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    print("2. Loading LoRA adapters (Retriever and Generator)...")
    retriever_lora_path = os.path.join(MODELS_OUT_DIR, f"lora_retriever_{safe_name}")
    generator_lora_path = os.path.join(MODELS_OUT_DIR, f"lora_generator_{safe_name}")

    model = PeftModel.from_pretrained(base_model, retriever_lora_path, adapter_name="retriever")
    model.load_adapter(generator_lora_path, adapter_name="generator")
    model.eval()

    test_file = os.path.join(DATASETS_DIR, "retriever_test.jsonl")
    records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    print(f"Loaded {len(records)} test records.")
    
    num_samples = min(5, len(records))
    samples = random.sample(records, num_samples)

    simulation_results = []

    for i, rec in enumerate(samples, 1):
        prompt_id = rec.get('prompt_id')
        print(f"\n\n{'#'*60}")
        print(f" SIMULATION EXAMPLE {i}/{num_samples} (Prompt ID: {prompt_id})")
        print(f"{'#'*60}")
        
        messages = rec["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        
        query_match = re.search(r"QUERY:\n(.*?)\n\nCONTEXT:", user_msg, re.DOTALL)
        context_match = re.search(r"CONTEXT:\n(.*)", user_msg, re.DOTALL)
        
        query = query_match.group(1).strip() if query_match else "BRAK QUERY"
        raw_context = context_match.group(1).strip() if context_match else "BRAK CONTEXTU"
        
        print(f"\n[USER QUERY]:\n{query}")
        
        # ---------------------------------------------------------
        # PHASE 1: RETRIEVER
        # ---------------------------------------------------------
        print("\n--- Sending data to retriever ---")
        model.set_adapter("retriever")
        
        retriever_user_prompt = f"QUERY:\n{query}\n\nCONTEXT:\n{raw_context}"
        retriever_output = generate_response(model, tokenizer, RETRIEVER_SYSTEM_PROMPT, retriever_user_prompt)
        
        ext_info = extract_tag(retriever_output, "extracted_information")
        ret_reasoning = extract_tag(retriever_output, "reasoning")
        is_sufficient = extract_tag(retriever_output, "is_sufficient")
        
        print(f"Status (is_sufficient): {is_sufficient}")
        print(f"Extracted data (JSON): {ext_info[:150]}... [ucięto]")

        # ---------------------------------------------------------
        # PHASE 2: GENERATOR
        # ---------------------------------------------------------
        print("\n--- Sending data to generator ---")
        model.set_adapter("generator")
        
        generator_user_prompt = f"QUERY: \n{query}\n\nCONTEXT:\n{raw_context}\n\nSUFFICIENCY:\n{is_sufficient}\n\nEXTRACTED:\n{ext_info}"
        generator_output = generate_response(model, tokenizer, GENERATOR_SYSTEM_PROMPT, generator_user_prompt)
        
        is_answerable = extract_tag(generator_output, "is_answerable")
        gen_reasoning = extract_tag(generator_output, "reasoning")
        hypothesis = extract_tag(generator_output, "hypothesis")
        natural_hypothesis = extract_tag(generator_output, "natural_hypothesis")
        
        print(f"Status (is_answerable): {is_answerable}")
        print(f"Reasoning:\n{gen_reasoning}")
        print(f"\nFinal Hypothesis (Strict):\n{hypothesis}")
        print(f"Final Hypothesis (Natural):\n{natural_hypothesis}")

        simulation_results.append({
            "prompt_id": prompt_id,
            "query": query,
            "retriever_is_sufficient": is_sufficient,
            "retriever_extracted_info": ext_info,
            "retriever_reasoning": ret_reasoning,
            "generator_is_answerable": is_answerable,
            "generator_reasoning": gen_reasoning,
            "generator_hypothesis_strict": hypothesis,
            "generator_hypothesis_natural": natural_hypothesis
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"pipeline_simulation_{timestamp}.csv"
    csv_filepath = os.path.join(EVAL_OUT_DIR, csv_filename)

    fieldnames = [
        "prompt_id", "query", 
        "retriever_is_sufficient", "retriever_reasoning", "retriever_extracted_info", 
        "generator_is_answerable", "generator_reasoning", "generator_hypothesis_strict", "generator_hypothesis_natural"
    ]

    with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(simulation_results)

    print(f"\n{'='*60}")
    print(f" COMPLETED. Simulation results saved to:\n {csv_filepath}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()