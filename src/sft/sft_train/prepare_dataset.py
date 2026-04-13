import os
import sys
import json
import random
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.dirname(SFT_DIR)
AGENTS_DIR = os.path.dirname(SRC_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.sft.utils.config import CONFIG

DATASETS_DIR = os.path.join(AGENTS_DIR, "data", "datasets")
OUTPUT_DATASET_DIR = os.path.join(AGENTS_DIR, CONFIG["paths"]["dataset_prepped"])
os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATASETS_DIR, CONFIG["files"].get("updated_sft_dataset", "multiagent_sft_dataset_v2.jsonl"))

RETRIEVER_TRAIN = os.path.join(OUTPUT_DATASET_DIR, "retriever_train.jsonl")
RETRIEVER_EVAL  = os.path.join(OUTPUT_DATASET_DIR, "retriever_eval.jsonl")
RETRIEVER_TEST  = os.path.join(OUTPUT_DATASET_DIR, "retriever_test.jsonl")

GENERATOR_TRAIN = os.path.join(OUTPUT_DATASET_DIR, "generator_train.jsonl")
GENERATOR_EVAL  = os.path.join(OUTPUT_DATASET_DIR, "generator_eval.jsonl")
GENERATOR_TEST  = os.path.join(OUTPUT_DATASET_DIR, "generator_test.jsonl")


RETRIEVER_SYSTEM_PROMPT = """You are an Expert Scientific Retriever Agent in a multi-agent system.
Your task is to analyze raw scientific context based on a user's research query.
You must extract the key variables, relationships, mechanisms, and evidence, and synthesize them into a concise, professional, natural-language message for the Generator agent.
Do NOT hallucinate or add any information outside of the provided context. Do NOT generate the final hypothesis yourself."""

GENERATOR_SYSTEM_PROMPT = """You are an AI Research Scientist agent.
You will receive a summary message from the Retriever agent containing extracted scientific data.
Your task is to formulate a strict, testable, causal hypothesis based ONLY on that data.

Format your response into two parts:
1. An internal reasoning block wrapped in <THOUGHT>...</THOUGHT> explaining if the data is sufficient and what the causal link is (or what is missing).
2. The final output:
   - If sufficient: Write the hypothesis directly as a continuous natural sentence.
   - If INSUFFICIENT: Write a direct request for specific missing information from the articles, wrapped in <REQUEST>...</REQUEST>.
   
Do NOT introduce new variables outside of what the Retriever provided."""

def create_chatml_record(prompt_id, system_msg, user_msg, assistant_msg):
    return {
        "prompt_id": prompt_id,
        "messages": [
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
    }

def run_tests(r_train, r_eval, r_test, g_train, g_eval, g_test):
    print("\n" + "=" * 50)
    print(" Running Dataset Integrity Tests...")
    print("=" * 50)

    r_train_ids = set(r["prompt_id"] for r in r_train)
    r_eval_ids  = set(r["prompt_id"] for r in r_eval)
    r_test_ids  = set(r["prompt_id"] for r in r_test)

    g_train_ids = set(r["prompt_id"] for r in g_train)
    g_eval_ids  = set(r["prompt_id"] for r in g_eval)
    g_test_ids  = set(r["prompt_id"] for r in g_test)

    tests_passed = True

    print("1. Checking for data leakage (retriever)...", end=" ")
    leakage_te = r_train_ids & r_eval_ids
    leakage_tt = r_train_ids & r_test_ids
    leakage_et = r_eval_ids  & r_test_ids
    if not any([leakage_te, leakage_tt, leakage_et]):
        print("OK!")
    else:
        print(f"\n   ERROR! Train∩Eval={leakage_te}, Train∩Test={leakage_tt}, Eval∩Test={leakage_et}")
        tests_passed = False

    print("2. Checking for data leakage (generator)...", end=" ")
    leakage_te = g_train_ids & g_eval_ids
    leakage_tt = g_train_ids & g_test_ids
    leakage_et = g_eval_ids  & g_test_ids
    if not any([leakage_te, leakage_tt, leakage_et]):
        print("OK!")
    else:
        print(f"\n   ERROR! Train∩Eval={leakage_te}, Train∩Test={leakage_tt}, Eval∩Test={leakage_et}")
        tests_passed = False

    print("3. Checking retriever/generator split alignment...", end=" ")
    if r_train_ids == g_train_ids and r_eval_ids == g_eval_ids and r_test_ids == g_test_ids:
        print("OK!")
    else:
        print("\n   ERROR! Generator IDs nie są identyczne z retriever IDs.")
        tests_passed = False

    print("4. Checking for empty required fields...", end=" ")
    empty_found = False
    for split_name, split in [("r_train", r_train), ("g_train", g_train)]:
        for rec in split:
            for msg in rec["messages"]:
                if not msg["content"].strip():
                    print(f"\n   ERROR! Empty content in {split_name}, "
                        f"prompt_id={rec['prompt_id']}, role={msg['role']}")
                    empty_found = True
                    tests_passed = False
    if not empty_found:
        print("OK!")

    print("-" * 50)
    if tests_passed:
        print("Result: All tests PASSED! Dataset is ready for training.")
    else:
        print("Result: Tests FAILED!")
        sys.exit(1)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found: {INPUT_FILE}")
        sys.exit(1)

    print(f"Loading data from: {INPUT_FILE}...")

    retriever_records = []
    generator_records = []
    
    prompt_combo = {}

    skipped_empty_context = 0
    stats = defaultdict(int)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            raw_context = data.get("raw_context", "").strip()
            if not raw_context:
                skipped_empty_context += 1
                continue

            prompt_id = data.get("prompt_id", "unknown")
            is_success = data.get("is_success", False)

            combo = f"success={is_success}"
            stats[combo] += 1
            prompt_combo[prompt_id] = combo

            # -------------------------
            # RETRIEVER RECORD
            # -------------------------
            retriever_user = f"USER QUERY:\n{data['user_query']}\n\nRAW CONTEXT:\n{raw_context}"

            retriever_assistant = data.get("retriever_message", "").strip()

            if len(retriever_assistant) < 10:
                continue

            retriever_records.append(
                create_chatml_record(
                    prompt_id,
                    RETRIEVER_SYSTEM_PROMPT,
                    retriever_user,
                    retriever_assistant,
                )
            )

            # -------------------------
            # GENERATOR RECORD
            # -------------------------
            generator_user = data.get("retriever_message", "")

            generator_assistant = data.get("generator_response", "").strip()

            if len(generator_assistant) < 20:
                continue

            if "<REQUEST>" in generator_assistant and len(generator_assistant) < 40:
                continue

            if generator_assistant.replace("<THOUGHT>", "").replace("</THOUGHT>", "").strip() == "":
                continue

            generator_records.append(
                create_chatml_record(
                    prompt_id,
                    GENERATOR_SYSTEM_PROMPT,
                    generator_user,
                    generator_assistant,
                )
            )

    print(f"\nSkipped (empty context): {skipped_empty_context}")
    print("\n=== Rozkład kombinacji klas ===")
    for combo, count in sorted(stats.items()):
        print(f"  {combo}: {count}")

    if len(retriever_records) == 0:
        print("ERROR: No valid records after filtering!")
        sys.exit(1)

    train_ids, eval_ids, test_ids = set(), set(), set()
    ids_by_combo = defaultdict(list)
    
    for pid, c in prompt_combo.items():
        ids_by_combo[c].append(pid)

    rng = random.Random(42)
    print("\n=== Stratyfikowany podział promptów ===")
    for c, pids in sorted(ids_by_combo.items()):
        rng.shuffle(pids)
        n = len(pids)

        n_train = max(1, int(n * 0.8)) if n >= 3 else (n if n > 0 else 0)
        n_eval  = max(1, int(n * 0.1)) if n >= 3 else 0
        n_test  = n - n_train - n_eval
        
        if n_test < 0:
            n_test = 0
            n_eval = max(0, n - n_train)

        train_ids.update(pids[:n_train])
        eval_ids.update(pids[n_train:n_train+n_eval])
        test_ids.update(pids[n_train+n_eval:])
        
        print(f"  Klasa '{c}': n={n} -> train={n_train}, eval={n_eval}, test={n_test}")

    r_train = [r for r in retriever_records if r["prompt_id"] in train_ids]
    r_eval  = [r for r in retriever_records if r["prompt_id"] in eval_ids]
    r_test  = [r for r in retriever_records if r["prompt_id"] in test_ids]

    g_train = [r for r in generator_records if r["prompt_id"] in train_ids]
    g_eval  = [r for r in generator_records if r["prompt_id"] in eval_ids]
    g_test  = [r for r in generator_records if r["prompt_id"] in test_ids]

    run_tests(r_train, r_eval, r_test, g_train, g_eval, g_test)

    def save_jsonl(records, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(records):>4} records -> {os.path.basename(filepath)}")

    print()
    save_jsonl(r_train, RETRIEVER_TRAIN)
    save_jsonl(r_eval,  RETRIEVER_EVAL)
    save_jsonl(r_test,  RETRIEVER_TEST)

    print()
    save_jsonl(g_train, GENERATOR_TRAIN)
    save_jsonl(g_eval,  GENERATOR_EVAL)
    save_jsonl(g_test,  GENERATOR_TEST)

    print("\nDataset ready for SFT.")

if __name__ == "__main__":
    main()