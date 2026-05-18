"""
prepare_dataset.py
==================
Prepares SFT dataset from GRPO-derived CSV files for unified Retriever+Generator LoRA.

Input files (placed in data/datasets/):
  - rl_retriever_outputs.csv
      columns: paper_id, paper_title, user_query, referenced_paper_ids,
               input_messages, reasoning, retriever_message

  - rl_generator_outputs_no_mask.csv
      columns: paper_id, paper_title, user_query, retriever_message,
               masked_retriever_message (empty), input_messages, reasoning, hypotheses

  - rl_generator_outputs_mask.csv
      same schema as above, but masked_retriever_message is filled

Output files (placed in data/datasets/sft3/):
  - shared_agent_train.jsonl
  - shared_agent_eval.jsonl    (only if EVAL_SPLIT_RATIO > 0)

Record format (ChatML with Qwen3 thinking):
  {
    "prompt_id": "<uid>",
    "agent_role": "retriever" | "generator",
    "messages": [
      {"role": "system",    "content": "<system prompt>"},
      {"role": "user",      "content": "<user turn>"},
      {"role": "assistant", "content": "<think>\\n{reasoning}\\n</think>\\n\\n{output}"}
    ]
  }

Notes
-----
- input_messages already contains the correct system+user turns from the GRPO
  pipeline so we reuse them directly — no prompt reconstruction needed.
- The assistant content is assembled as:
      <think>\n{reasoning}\n</think>\n\n{output}
  This matches the Qwen3 thinking format expected by the tokenizer and keeps
  the reasoning inside the training loss (we want the model to learn to reason).
- masked_retriever_message records are included as a data-augmentation variant:
  the generator sees a noisy channel message and still has to produce valid
  hypotheses. They are tagged agent_role="generator_masked" so you can filter
  them out if you change your mind.
- No separate test split is created — you only need train + eval for SFT when
  the primary evaluation is the downstream GRPO comparison.
"""

import ast
import json
import os
import random
import sys
import uuid
from collections import defaultdict

import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_DIR    = os.path.dirname(SCRIPT_DIR)
SRC_DIR    = os.path.dirname(SFT_DIR)
AGENTS_DIR = os.path.dirname(SRC_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.sft.utils.config import CONFIG

DATASETS_DIR       = os.path.join(AGENTS_DIR, "data", "datasets", "sft_raw")
OUTPUT_DATASET_DIR = os.path.join(AGENTS_DIR, CONFIG["paths"]["dataset_prepped"])
os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

# ── input files ───────────────────────────────────────────────────────────────

RETRIEVER_CSV        = os.path.join(DATASETS_DIR, "rl_retriever_outputs.csv")
GENERATOR_NO_MASK_CSV = os.path.join(DATASETS_DIR, "rl_generator_outputs_no_mask.csv")
GENERATOR_MASK_CSV   = os.path.join(DATASETS_DIR, "rl_generator_outputs_mask.csv")

# ── output files ──────────────────────────────────────────────────────────────

TRAIN_FILE = os.path.join(OUTPUT_DATASET_DIR, "shared_agent_train.jsonl")
EVAL_FILE  = os.path.join(OUTPUT_DATASET_DIR, "shared_agent_eval.jsonl")

# ── split config ──────────────────────────────────────────────────────────────

EVAL_SPLIT_RATIO = 0.1   # set to 0.0 to skip eval split entirely
RANDOM_SEED      = 42

# ── include masked generator records as augmentation? ────────────────────────

USE_MASKED_AUGMENTATION = True


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_messages(raw) -> list:
    """Parse input_messages from CSV cell (JSON string or Python list literal)."""
    if isinstance(raw, list):
        return raw
    raw = str(raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: some CSVs are saved with Python repr instead of JSON
        return ast.literal_eval(raw)


def _build_assistant_content(reasoning: str, output: str) -> str:
    """
    Assemble the assistant turn in Qwen3 thinking format:

        <think>
        {reasoning}
        </think>

        {output}

    If reasoning is empty/NaN we still emit an empty think block so the
    template stays consistent and the model learns when NOT to reason.
    """
    reasoning = str(reasoning).strip() if (reasoning and str(reasoning) != "nan") else ""
    output    = str(output).strip()
    return f"<think>\n{reasoning}\n</think>\n\n{output}"


def _make_record(
    prompt_id: str,
    agent_role: str,
    messages: list,        # system + user turns from input_messages
    reasoning: str,
    output: str,
) -> dict:
    """Create a single ChatML training record."""
    assistant_content = _build_assistant_content(reasoning, output)
    full_messages = list(messages) + [
        {"role": "assistant", "content": assistant_content}
    ]
    return {
        "prompt_id": prompt_id,
        "agent_role": agent_role,
        "messages": full_messages,
    }


# ─────────────────────────────────────────────────────────────────────────────
# loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_retriever_records(csv_path: str) -> list:
    """
    Load retriever SFT records.

    Each row becomes one record where:
      - system + user  → from input_messages (already built by GRPO pipeline)
      - assistant      → <think>{reasoning}</think>\\n\\n{retriever_message}
    """
    df = pd.read_csv(csv_path)
    required = {"input_messages", "reasoning", "retriever_message"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"rl_retriever_outputs.csv is missing columns: {missing}")

    records = []
    skipped = 0

    for _, row in df.iterrows():
        retriever_msg = str(row.get("retriever_message", "")).strip()
        if not retriever_msg or retriever_msg == "nan":
            skipped += 1
            continue

        try:
            messages = _parse_messages(row["input_messages"])
        except Exception as e:
            print(f"  [WARN] Could not parse input_messages for retriever row: {e}")
            skipped += 1
            continue

        # Keep only system + user turns (drop any existing assistant turn)
        sys_user = [m for m in messages if m["role"] in ("system", "user")]
        if len(sys_user) < 2:
            skipped += 1
            continue

        pid = str(row.get("paper_id", uuid.uuid4().hex))
        records.append(
            _make_record(
                prompt_id  = pid,
                agent_role = "retriever",
                messages   = sys_user,
                reasoning  = row.get("reasoning", ""),
                output     = retriever_msg,
            )
        )

    print(f"  Retriever records loaded : {len(records)}  (skipped: {skipped})")
    return records


def load_generator_records(csv_path: str, agent_role: str, output_col: str) -> list:
    """
    Load generator SFT records from a CSV file.

    Parameters
    ----------
    csv_path   : path to the CSV file
    agent_role : "generator" or "generator_masked"
    output_col : column that holds the target output
                 ("hypotheses" for no_mask, "hypotheses" for mask too — the
                 difference is only in the user turn via masked_retriever_message
                 already embedded in input_messages for the mask file)
    """
    df = pd.read_csv(csv_path)
    required = {"input_messages", "reasoning", output_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(csv_path)} is missing columns: {missing}")

    records = []
    skipped = 0

    for _, row in df.iterrows():
        output_text = str(row.get(output_col, "")).strip()
        if not output_text or output_text == "nan" or len(output_text) < 20:
            skipped += 1
            continue

        # Basic format check: generator output must contain at least one numbered
        # hypothesis line (e.g. "1. ..."). Silently drop malformed rows.
        import re
        if not re.search(r"(?m)^\s*\d+[.)]\s+\S", output_text):
            skipped += 1
            continue

        try:
            messages = _parse_messages(row["input_messages"])
        except Exception as e:
            print(f"  [WARN] Could not parse input_messages for generator row: {e}")
            skipped += 1
            continue

        sys_user = [m for m in messages if m["role"] in ("system", "user")]
        if len(sys_user) < 2:
            skipped += 1
            continue

        pid = str(row.get("paper_id", uuid.uuid4().hex))
        records.append(
            _make_record(
                prompt_id  = pid,
                agent_role = agent_role,
                messages   = sys_user,
                reasoning  = row.get("reasoning", ""),
                output     = output_text,
            )
        )

    print(f"  Generator ({agent_role}) records loaded: {len(records)}  (skipped: {skipped})")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# integrity tests
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(train: list, eval_: list) -> None:
    print("\n" + "=" * 55)
    print(" Running Dataset Integrity Tests")
    print("=" * 55)
    passed = True

    # 1. no data leakage between splits
    if eval_:
        train_ids = {r["prompt_id"] for r in train}
        eval_ids  = {r["prompt_id"] for r in eval_}
        overlap   = train_ids & eval_ids
        print(f"Data leakage check ... ", end="")
        if not overlap:
            print("OK")
        else:
            print(f"WARN — {len(overlap)} shared prompt_ids (expected for paper-level split).")
            # Not fatal: same paper_id can appear in both splits but with different
            # agent_role records; only truly problematic if the exact same record appears.
            exact_train = {(r["prompt_id"], r["agent_role"]) for r in train}
            exact_eval  = {(r["prompt_id"], r["agent_role"]) for r in eval_}
            exact_overlap = exact_train & exact_eval
            if exact_overlap:
                print(f"  ERROR — {len(exact_overlap)} exact (prompt_id, agent_role) duplicates!")
                passed = False

    # 2. no empty fields
    print("Empty content check  ... ", end="")
    empty_found = False
    for split_name, split in [("train", train), ("eval", eval_)]:
        for rec in split:
            for msg in rec["messages"]:
                if not str(msg.get("content", "")).strip():
                    print(f"\n  ERROR — empty content in {split_name}, "
                          f"prompt_id={rec['prompt_id']}, role={msg['role']}")
                    empty_found = True
                    passed = False
    if not empty_found:
        print("OK")

    # 3. assistant turn format
    print("Thinking format check ... ", end="")
    fmt_errors = 0
    for rec in train + eval_:
        asst = next((m for m in rec["messages"] if m["role"] == "assistant"), None)
        if asst and "<think>" not in asst["content"]:
            fmt_errors += 1
    if fmt_errors == 0:
        print("OK")
    else:
        print(f"WARN — {fmt_errors} assistant turns missing <think> block")

    # 4. role distribution
    role_counts = defaultdict(int)
    for rec in train + eval_:
        role_counts[rec["agent_role"]] += 1
    print("\nAgent role distribution (train+eval):")
    for role, cnt in sorted(role_counts.items()):
        print(f"  {role:25s}: {cnt}")

    print("-" * 55)
    if passed:
        print("Result: All tests PASSED — dataset ready for SFT.")
    else:
        print("Result: FAILED — fix errors above before training.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for path, label in [
        (RETRIEVER_CSV,         "rl_retriever_outputs.csv"),
        (GENERATOR_NO_MASK_CSV, "rl_generator_outputs_no_mask.csv"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: Required file not found: {path}")
            sys.exit(1)

    print("Loading data...\n")

    # ── load all records ──────────────────────────────────────────────────────
    retriever_records = load_retriever_records(RETRIEVER_CSV)
    generator_records = load_generator_records(
        GENERATOR_NO_MASK_CSV, agent_role="generator", output_col="hypotheses"
    )

    masked_records = []
    if USE_MASKED_AUGMENTATION and os.path.exists(GENERATOR_MASK_CSV):
        masked_records = load_generator_records(
            GENERATOR_MASK_CSV, agent_role="generator_masked", output_col="hypotheses"
        )
    elif USE_MASKED_AUGMENTATION:
        print(f"  [WARN] Mask file not found at {GENERATOR_MASK_CSV} — skipping augmentation.")

    all_records = retriever_records + generator_records + masked_records

    if not all_records:
        print("ERROR: No valid records after filtering.")
        sys.exit(1)

    print(f"\nTotal records before split: {len(all_records)}")

    # ── split ─────────────────────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)

    if EVAL_SPLIT_RATIO <= 0.0:
        rng.shuffle(all_records)
        train_records = all_records
        eval_records  = []
        print(f"No eval split (EVAL_SPLIT_RATIO=0) — all {len(train_records)} records go to train.")
    else:
        # Stratify by agent_role so each split has a balanced role mix
        by_role: dict = defaultdict(list)
        for rec in all_records:
            by_role[rec["agent_role"]].append(rec)

        train_records, eval_records = [], []
        print("\n=== Stratified split by agent_role ===")
        for role, recs in sorted(by_role.items()):
            rng.shuffle(recs)
            n_eval  = max(1, int(len(recs) * EVAL_SPLIT_RATIO))
            n_train = len(recs) - n_eval
            train_records.extend(recs[:n_train])
            eval_records.extend(recs[n_train:])
            print(f"  {role:25s}: total={len(recs):4d}  train={n_train:4d}  eval={n_eval:4d}")

        rng.shuffle(train_records)
        rng.shuffle(eval_records)

    # ── integrity tests ───────────────────────────────────────────────────────
    run_tests(train_records, eval_records)

    # ── save ──────────────────────────────────────────────────────────────────
    def save_jsonl(records: list, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(records):>5} records → {os.path.basename(path)}")

    print("\n=== Saving ===")
    save_jsonl(train_records, TRAIN_FILE)
    if eval_records:
        save_jsonl(eval_records, EVAL_FILE)

    print("\nDataset ready for SFT.")


if __name__ == "__main__":
    main()