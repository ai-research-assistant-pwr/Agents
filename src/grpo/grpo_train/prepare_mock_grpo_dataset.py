import os
import sys
import json
import random
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRPO_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.dirname(GRPO_DIR)
AGENTS_DIR = os.path.dirname(SRC_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

DATASETS_DIR = os.path.join(AGENTS_DIR, "data", "datasets")
SFT_TEST_FILE = os.path.join(DATASETS_DIR, "sft2", "generator_test.jsonl")
SOURCE_DATASET_FILE = os.path.join(DATASETS_DIR, "multiagent_sft_dataset_v2.jsonl")

OUTPUT_DIR = os.path.join(DATASETS_DIR, "grpo_exp_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mock_data.json")

def main():
    print(f"=== Preparing GRPO Experiment Dataset ===")
    
    if not os.path.exists(SFT_TEST_FILE):
        print(f"ERROR: File not found: {SFT_TEST_FILE}")
        sys.exit(1)
        
    if not os.path.exists(SOURCE_DATASET_FILE):
        print(f"ERROR: File not found: {SOURCE_DATASET_FILE}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_ids = set()
    with open(SFT_TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            test_ids.add(record.get("prompt_id"))
            
    print(f"Found {len(test_ids)} unique prompt_id in the SFT test set.")

    success_records = []
    fail_records = []
    
    GENERATOR_SYSTEM_PROMPT = """You are an AI Research Scientist agent.
You will receive a summary message from the Retriever agent containing extracted scientific data.
Your task is to formulate a strict, testable, causal hypothesis based ONLY on that data.

Format your response into two parts:
1. An internal reasoning block wrapped in <THOUGHT>...</THOUGHT> explaining if the data is sufficient and what the causal link is (or what is missing).
2. The final output:
   - If sufficient: Write the hypothesis directly as a continuous natural sentence.
   - If INSUFFICIENT: Write a direct request for specific missing information from the articles, wrapped in <REQUEST>...</REQUEST>.
   
Do NOT introduce new variables outside of what the Retriever provided."""

    with open(SOURCE_DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            prompt_id = data.get("prompt_id")
            
            if prompt_id in test_ids:
                is_success = data.get("is_success", False)
                retriever_message = data.get("retriever_message", "").strip()
                
                hidden_chunks = []
                if not is_success:
                    hidden_chunks = [
                        "ADDITIONAL CONTEXT: Received new detailed information that may be relevant to the query.",
                        "Variable X has a direct impact on Y through mechanism Z."
                    ]
                
                hidden_data_str = json.dumps(hidden_chunks)
                
                full_prompt = textwrap.dedent(f"""\
                <HIDDEN_CHUNKS>{hidden_data_str}</HIDDEN_CHUNKS>
                <|im_start|>system
                {GENERATOR_SYSTEM_PROMPT}

                IMPORTANT:
                - Use <THOUGHT>...</THOUGHT> for reasoning
                - Use <REQUEST>...</REQUEST> if more data is needed
                <|im_end|>
                <|im_start|>user
                {retriever_message}
                <|im_end|>
                <|im_start|>assistant
                """)
                
                grpo_record = {
                    "id": prompt_id,
                    "query": full_prompt,
                    "expected_action": "GENERATE" if is_success else "ASK",
                    "label": "GENERATE" if is_success else "ASK"
                }
                
                if is_success:
                    success_records.append(grpo_record)
                else:
                    fail_records.append(grpo_record)

    random.seed(42)
    random.shuffle(success_records)
    random.shuffle(fail_records)

    sampled_success = success_records[:25]
    sampled_fail = fail_records[:25]

    print(f"Selected {len(sampled_success)} 'GENERATE' records and {len(sampled_fail)} 'ASK' records.")

    final_records = sampled_success + sampled_fail
    random.shuffle(final_records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(final_records)} mixed records to file: {OUTPUT_FILE}")
    print("=== Completed successfully ===")

if __name__ == "__main__":
    main()