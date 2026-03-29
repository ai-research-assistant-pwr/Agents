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

INPUT_FILE = os.path.join(DATASETS_DIR, CONFIG["files"]["synthetic_sft_dataset"])

RETRIEVER_TRAIN = os.path.join(OUTPUT_DATASET_DIR, "retriever_train.jsonl")
RETRIEVER_EVAL  = os.path.join(OUTPUT_DATASET_DIR, "retriever_eval.jsonl")
RETRIEVER_TEST  = os.path.join(OUTPUT_DATASET_DIR, "retriever_test.jsonl")

GENERATOR_TRAIN = os.path.join(OUTPUT_DATASET_DIR, "generator_train.jsonl")
GENERATOR_EVAL  = os.path.join(OUTPUT_DATASET_DIR, "generator_eval.jsonl")
GENERATOR_TEST  = os.path.join(OUTPUT_DATASET_DIR, "generator_test.jsonl")

# Wiadomość dla generatora gdy kontekst jest niewystarczający.
# Używana jako hypothesis_statement w negative cases zamiast pustego stringa.
GENERATOR_INSUFFICIENT_MSG = (
    "A valid hypothesis cannot be generated because the provided context "
    "does not contain sufficient variables or relationships to support "
    "a grounded causal claim relevant to the query."
)

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

    # Test 1 — leakage między splitami (retriever)
    print("1. Checking for data leakage (retriever)...", end=" ")
    leakage_te = r_train_ids & r_eval_ids
    leakage_tt = r_train_ids & r_test_ids
    leakage_et = r_eval_ids  & r_test_ids
    if not any([leakage_te, leakage_tt, leakage_et]):
        print("OK!")
    else:
        print(f"\n   ERROR! Train∩Eval={leakage_te}, Train∩Test={leakage_tt}, Eval∩Test={leakage_et}")
        tests_passed = False

    # Test 2 — leakage między splitami (generator)
    print("2. Checking for data leakage (generator)...", end=" ")
    leakage_te = g_train_ids & g_eval_ids
    leakage_tt = g_train_ids & g_test_ids
    leakage_et = g_eval_ids  & g_test_ids
    if not any([leakage_te, leakage_tt, leakage_et]):
        print("OK!")
    else:
        print(f"\n   ERROR! Train∩Eval={leakage_te}, Train∩Test={leakage_tt}, Eval∩Test={leakage_et}")
        tests_passed = False

    # Test 3 — retriever i generator mają te same ID w każdym splicie
    print("3. Checking retriever/generator split alignment...", end=" ")
    if r_train_ids == g_train_ids and r_eval_ids == g_eval_ids and r_test_ids == g_test_ids:
        print("OK!")
    else:
        # Generator może mieć mniej rekordów niż retriever (inne filtrowanie),
        # więc tu sprawdzamy tylko że generator IDs są podzbiorem retriever IDs
        if g_train_ids.issubset(r_train_ids) and g_eval_ids.issubset(r_eval_ids) and g_test_ids.issubset(r_test_ids):
            print("OK (generator is subset of retriever — expected).")
        else:
            print("\n   ERROR! Generator IDs nie są podzbiorem retriever IDs w którymś splicie.")
            tests_passed = False

    # Test 4 — brak pustych pól w rekordach treningowych
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
    
    # Przechowujemy kombinację klas dla każdego promptu do poprawnej stratyfikacji
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
            is_sufficient = data.get("retriever_is_sufficient", False)
            is_answerable = data.get("generator_is_answerable", False)

            # Identyfikujemy "profil" tego promptu
            combo = f"ret={is_sufficient}/gen={is_answerable}"
            stats[combo] += 1
            prompt_combo[prompt_id] = combo

            # -------------------------
            # RETRIEVER RECORD
            # -------------------------
            retriever_user = f"""QUERY:
{data['user_query']}

CONTEXT:
{raw_context}"""

            retriever_assistant = f"""<extracted_information>
{data['retriever_extracted_info']}
</extracted_information>

<reasoning>
{data['retriever_reasoning']}
</reasoning>

<is_sufficient>
{data['retriever_is_sufficient']}
</is_sufficient>"""

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
            if not is_answerable and not is_sufficient:
                # Pomijamy: ret=False/gen=False trafia tylko do retriever dataset
                continue

            generator_user = f"""QUERY: 
{data['user_query']}

CONTEXT:
{raw_context}

SUFFICIENCY:
{data['retriever_is_sufficient']}

EXTRACTED:
{data['retriever_extracted_info']}"""

            if is_answerable:
                gen_hyp = data.get("generator_hypothesis", "").strip()
                gen_nat = data.get("generator_natural_hypothesis", "").strip()
                gen_fal = data.get("generator_falsification", "").strip()

                generator_assistant = f"""<is_answerable>
{data['generator_is_answerable']}
</is_answerable>

<reasoning>
{data['generator_reasoning']}
</reasoning>

<hypothesis>
{gen_hyp}
</hypothesis>

<natural_hypothesis>
{gen_nat}
</natural_hypothesis>

<falsification_criteria>
{gen_fal}
</falsification_criteria>"""
            else:
                generator_assistant = f"""<is_answerable>
False
</is_answerable>

<reasoning>
{data['generator_reasoning']}
</reasoning>

<hypothesis>
{GENERATOR_INSUFFICIENT_MSG}
</hypothesis>

<natural_hypothesis>
</natural_hypothesis>

<falsification_criteria>
</falsification_criteria>"""

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
        
        # Obliczanie proporcji 80/10/10
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

    # Przypisywanie wygenerowanych rekordów do splitów na podstawie ich prompt_id
    r_train = [r for r in retriever_records if r["prompt_id"] in train_ids]
    r_eval  = [r for r in retriever_records if r["prompt_id"] in eval_ids]
    r_test  = [r for r in retriever_records if r["prompt_id"] in test_ids]

    g_train = [r for r in generator_records if r["prompt_id"] in train_ids]
    g_eval  = [r for r in generator_records if r["prompt_id"] in eval_ids]
    g_test  = [r for r in generator_records if r["prompt_id"] in test_ids]

    # Uruchomienie rygorystycznych testów
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

