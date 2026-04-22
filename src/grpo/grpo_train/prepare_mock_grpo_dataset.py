import os
import sys
import json
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRPO_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.dirname(GRPO_DIR)
AGENTS_DIR = os.path.dirname(SRC_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.grpo.grpo_train.agents import AgentPrompts

DATASETS_DIR = os.path.join(AGENTS_DIR, "data", "datasets")
SFT_TEST_FILE = os.path.join(DATASETS_DIR, "sft2", "generator_test.jsonl")
SOURCE_DATASET_FILE = os.path.join(DATASETS_DIR, "multiagent_sft_dataset_v2.jsonl")

OUTPUT_DIR = os.path.join(DATASETS_DIR, "grpo_exp_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mock_data.json")

def main():
    print(f"=== Preparing Multi-Agent GRPO Experiment Dataset ===")
    
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
    
    retriever_system_prompt = AgentPrompts.get_retriever_system_prompt()

    with open(SOURCE_DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            prompt_id = data.get("prompt_id")
            
            if prompt_id in test_ids:
                is_success = data.get("is_success", False)
                initial_chunks = [
                    "Study shows correlation between A and B.",
                    "Mechanisms remain unclear, but preliminary data suggests pathway C."
                ]
                initial_data_str = json.dumps(initial_chunks)
                
                hidden_chunks = []
                if not is_success:
                    hidden_chunks = [
                        "ADDITIONAL CONTEXT: Pathway C is activated by enzyme D.",
                        "Variable X has a direct impact on Y through mechanism Z."
                    ]
                hidden_data_str = json.dumps(hidden_chunks)
                full_prompt = (
                    f"<HIDDEN_CHUNKS>{hidden_data_str}</HIDDEN_CHUNKS>\n"
                    f"<|im_start|>system\n"
                    f"{retriever_system_prompt}\n"
                    f"<|im_end|>\n"
                    f"<|im_start|>user\n"
                    f"RAW CHUNKS:\n{initial_data_str}\n"
                    f"<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                
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