import os
import sys
import json
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.dirname(SFT_DIR)
AGENTS_DIR = os.path.dirname(SRC_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.sft.utils.config import CONFIG

DATASETS_DIR = os.path.join(AGENTS_DIR, "data", "datasets")
OUTPUT_DATASET_DIR = os.path.join(DATASETS_DIR, CONFIG["files"]["dataset_prepped"])
os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATASETS_DIR, CONFIG["paths"]["synthetic_sft_dataset"])

RETRIEVER_TRAIN = os.path.join(DATASETS_DIR, "retriever_train.jsonl")
RETRIEVER_EVAL = os.path.join(DATASETS_DIR, "retriever_eval.jsonl")
GENERATOR_TRAIN = os.path.join(DATASETS_DIR, "generator_train.jsonl")
GENERATOR_EVAL = os.path.join(DATASETS_DIR, "generator_eval.jsonl")

RETRIEVER_SYSTEM_PROMPT = """You are an Expert Scientific Retriever Agent.

Your job is to analyze raw scientific context and prepare structured knowledge for hypothesis generation.

Your output MUST be formatted EXACTLY using the following XML tags:

<reasoning>
- Step-by-step explanation of how the context relates to the query.
- Identify relevant vs irrelevant parts.
- Explain mechanisms and relationships.
</reasoning>

<extracted_information>
- Structured extraction of key variables, relationships, mechanisms, and data points relevant to the query.
</extracted_information>

IMPORTANT:
- Do NOT generate hypotheses.
- If context is insufficient, clearly state it in reasoning and extracted information."""

GENERATOR_SYSTEM_PROMPT = """You are an AI Research Scientist generating scientific hypotheses.

You will receive:
- Research Query
- Raw Context
- Retriever Sufficiency
- Retriever Reasoning
- Extracted Information

Your output MUST be formatted EXACTLY using the following XML tags:

<is_answerable>
True ONLY if the retriever output contains enough grounded information. (Answer with True or False)
</is_answerable>

<reasoning>
- Step-by-step scientific reasoning.
- Explain how extracted information leads to the hypothesis.
</reasoning>

<hypothesis>
- Write it in a natural, highly professional academic style.
- Clearly state the proposed relationships, effects, or mechanisms.
</hypothesis>

<natural_hypothesis>
- Same hypothesis written in natural academic style
</natural_hypothesis>

<falsification_criteria>
- A specific, measurable experimental result or condition that would prove the hypothesis WRONG.
</falsification_criteria>

IMPORTANT:
- Hypothesis must be grounded in extracted information.
- Do NOT introduce new variables.
- If information is insufficient, set <is_answerable>False</is_answerable>.
- If RETRIEVER SUFFICIENCY is False, you MUST set <is_answerable>False</is_answerable>.  # 🔥 FIX
"""


def create_chatml_record(prompt_id, system_msg, user_msg, assistant_msg):
    return {
        "prompt_id": prompt_id,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
    }


def run_tests(r_train, r_eval, g_train, g_eval):
    print("\n" + "=" * 50)
    print(" Running Dataset Integrity Tests...")
    print("=" * 50)

    # ID extraction for leakage and alignment tests
    r_train_ids = set(r["prompt_id"] for r in r_train)
    r_eval_ids = set(r["prompt_id"] for r in r_eval)
    g_train_ids = set(r["prompt_id"] for r in g_train)
    g_eval_ids = set(r["prompt_id"] for r in g_eval)

    tests_passed = True

    # Test 1: Record counts
    print("1. Checking record counts (Train/Eval)...", end=" ")
    if len(r_train) == len(g_train) and len(r_eval) == len(g_eval):
        print("OK!")
    else:
        print(f"\n   ERROR! R_train: {len(r_train)}, G_train: {len(g_train)}")
        tests_passed = False

    # Test 2: Data leakage check
    print("2. Checking for data leakage (Data Leakage)...", end=" ")
    leakage = r_train_ids.intersection(r_eval_ids)
    if not leakage:
        print("OK!")
    else:
        print(f"\n   ERROR! Found common IDs in Train and Eval: {leakage}")
        tests_passed = False

    # Test 3: Prompt alignment check
    print("3. Checking prompt alignment (Alignment)...", end=" ")
    if r_train_ids == g_train_ids and r_eval_ids == g_eval_ids:
        print("OK!")
    else:
        print("\n   ERROR! Generator and Retriever have different sets of IDs!")
        tests_passed = False

    print("-" * 50)
    if tests_passed:
        print("Result: All tests PASSED! Dataset is ready for training.")
    else:
        print("Result: Tests FAILED! Check the dataset partitioning logic.")
        sys.exit(1)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found: {INPUT_FILE}")
        sys.exit(1)

    print(f"Loading data from: {INPUT_FILE}...")

    retriever_records = []
    generator_records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            prompt_id = data.get("prompt_id", "unknown")

            # =========================
            # RETRIEVER
            # =========================
            retriever_user = f"""RESEARCH QUERY:
{data['user_query']}

RAW CONTEXT:
{data['raw_context']}"""

            retriever_assistant = f"""<reasoning>
{data['retriever_reasoning']}
</reasoning>

<extracted_information>
{data['retriever_extracted_info']}
</extracted_information>"""

            retriever_records.append(
                create_chatml_record(
                    prompt_id,
                    RETRIEVER_SYSTEM_PROMPT,
                    retriever_user,
                    retriever_assistant,
                )
            )

            # =========================
            # GENERATOR
            # =========================
            generator_user = f"""RESEARCH QUERY:
{data['user_query']}

RAW CONTEXT:
{data['raw_context']}

RETRIEVER SUFFICIENCY:
{data['retriever_is_sufficient']}

RETRIEVER REASONING:
{data['retriever_reasoning']}

EXTRACTED INFORMATION:
{data['retriever_extracted_info']}""" 

            generator_assistant = f"""<is_answerable>
{data['generator_is_answerable']}
</is_answerable>

<reasoning>
{data['generator_reasoning']}
</reasoning>

<hypothesis>
{data['generator_hypothesis']}
</hypothesis>

<natural_hypothesis>
{data['generator_natural_hypothesis']}
</natural_hypothesis>

<falsification_criteria>
{data['generator_falsification']}
</falsification_criteria>"""

            generator_records.append(
                create_chatml_record(
                    prompt_id,
                    GENERATOR_SYSTEM_PROMPT,
                    generator_user,
                    generator_assistant,
                )
            )

    print(f"Processed {len(retriever_records)} records.")

    combined = list(zip(retriever_records, generator_records))
    random.seed(42)
    random.shuffle(combined)

    retriever_records, generator_records = zip(*combined)
    retriever_records = list(retriever_records)
    generator_records = list(generator_records)

    split_idx = int(len(retriever_records) * 0.9)

    r_train, r_eval = retriever_records[:split_idx], retriever_records[split_idx:]
    g_train, g_eval = generator_records[:split_idx], generator_records[split_idx:]

    run_tests(r_train, r_eval, g_train, g_eval)

    def save_jsonl(records, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(records)} -> {os.path.basename(filepath)}")

    save_jsonl(r_train, RETRIEVER_TRAIN)
    save_jsonl(r_eval, RETRIEVER_EVAL)
    save_jsonl(g_train, GENERATOR_TRAIN)
    save_jsonl(g_eval, GENERATOR_EVAL)

    print("\nDataset ready for SFT.")


if __name__ == "__main__":
    main()
