"""
Run the hypothesis generator model (Qwen3-32B via vLLM) over retriever outputs.

For each row in the retriever output CSV the script:
  1. Reads the user_query and retriever_message.
  2. Optionally applies span masking to the retriever_message (--use-masking).
  3. Builds a messages list (system + user) mirroring _build_generator_generate_prompt.
  4. Calls the model and saves full input and output (reasoning + content) to a CSV.

Prerequisites – start the vLLM server first:
    vllm serve Qwen/Qwen3-32B \\
      --tensor-parallel-size 2 \\
      --reasoning-parser qwen3

Usage:
    python scripts/rl/run_generator_inference.py \\
        --retriever-out data/rl_retriever_outputs.csv \\
        --out data/rl_generator_outputs.csv

    python scripts/rl/run_generator_inference.py \\
        --retriever-out data/rl_retriever_outputs.csv \\
        --out data/rl_generator_outputs.csv \\
        --use-masking \\
        --mask-prob 0.15 \\
        --span-prob 0.2 \\
        --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen3-32B \\
        --workers 8 \\
        --n 100
"""

import argparse
import csv
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]

csv.field_size_limit(10_000_000)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM = (
    "You are an Expert Research Scientist specializing in scientific hypothesis "
    "generation. You have received evidence syntheses from a retriever agent. "
    "Generate high-quality scientific hypotheses based on this evidence.\n\n"
    "## Hypothesis Quality Criteria\n\n"
    "Every hypothesis MUST satisfy ALL of the following:\n\n"
    "1. **Clear and precisely stated** with well-defined variables and a proposed "
    "relationship or mechanism.\n"
    "2. **Directly relevant** to the research query.\n"
    "3. **Grounded in the provided evidence** — traceable to specific findings "
    "in the retriever's syntheses.\n"
    "4. **Novel** — synthesize across findings, propose mechanistic explanations, "
    "or suggest extensions to new conditions.\n"
    "5. **Testable and falsifiable** — a conceivable experiment could confirm or refute it.\n"
    "6. **Multi-sentence** — describe the relationship, the proposed mechanism, and "
    "the conditions under which it holds. Split complex ideas across multiple sentences "
    "to improve clarity.\n"
    "7. **Self-contained** — every newly introduced term, dataset, or method that is "
    "essential to understanding the hypothesis must be explained within the hypothesis "
    "itself. A domain expert reading only the hypothesis statement should be able to "
    "design an experiment to test it without needing additional context. After drafting "
    "each hypothesis, ask yourself: 'If I only had this statement, could I design an "
    "experiment to test it?' If the answer is no, revise to include the necessary "
    "clarifications.\n"
    "8. **Specific, not generic** — avoid hypotheses that are too general or obvious. "
    "Focus on specific, non-trivial claims about mechanisms, interactions, or conditions.\n"
    "9. **Method-focused when proposing new approaches** — if a hypothesis proposes a "
    "novel methodology or technique, it should describe the method itself in detail "
    "(what it does, how it integrates existing techniques, why it is expected to work), "
    "not just state that it achieves better performance on some task. For example, a "
    "good hypothesis would be: 'We propose method X that integrates techniques A and B "
    "in a unique way, allowing it to effectively leverage both structured and unstructured "
    "data for improved performance on task Y.'\n\n"
    "Fewer well-grounded hypotheses are better than many speculative ones.\n\n"
    "## Handling Channel Noise and [MASK] Tokens\n"
    "The synthesis from the retriever agent may contain channel noise, specifically "
    "corrupted or missing words represented by the `[MASK]` token.\n"
    "- **CRITICAL:** Do NOT attempt to guess, reconstruct, or fill in these missing words.\n"
    "- Do NOT mention the `[MASK]` tokens, text corruption, or missing text in your thoughts or output.\n"
    "- Treat `[MASK]` purely as irrelevant background noise. Focus your analytical "
    "attention entirely on the intact, readable scientific concepts and data surrounding the masks.\n\n"
    "## Output Format\n\n"
    "Output ONLY a numbered list of hypotheses, one per line, like:\n"
    "1. [First hypothesis text]\n"
    "2. [Second hypothesis text]\n"
    "3. [Third hypothesis text]\n\n"
    "Write only the hypothesis text — no preamble, no commentary, no leading phrases "
    "such as 'The hypothesis is:' or 'This paper proposes:'. Each hypothesis must stand "
    "on its own as a clean, concise declarative statement."
)

# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def apply_span_masking(
    text: str,
    rng: random.Random,
    mask_prob: float = 0.15,
    span_prob: float = 0.2,
    min_word_len: int = 3,
) -> str:
    """
    Apply span masking to *text*:

    1. Tokenise on whitespace (preserving position).
    2. Identify eligible tokens: those whose alphabetic characters total at
       least *min_word_len* letters.
    3. Each eligible token is independently selected for masking with
       probability *mask_prob*.
    4. For each selected token, with probability *span_prob* extend the mask
       to cover 1–3 additional consecutive tokens (span masking).
    5. Replace every masked token with ``[MASK]``.
    """
    # Split preserving whitespace segments between tokens so we can reconstruct.
    # Use re.split with a capturing group to keep the separators.
    parts = re.split(r"(\s+)", text)
    # parts is interleaved: [token, sep, token, sep, ..., token]
    # Indices of actual tokens (non-separator parts):
    token_indices = [
        i for i, p in enumerate(parts) if not re.fullmatch(r"\s*", p) or (i % 2 == 0)
    ]
    # Simpler: tokens are at even indices if text doesn't start with whitespace,
    # but that's fragile. Let's just identify non-whitespace segments.
    token_positions = [
        i for i, p in enumerate(parts) if p and not re.fullmatch(r"\s+", p)
    ]

    masked = set()

    i = 0
    while i < len(token_positions):
        pos = token_positions[i]
        word = parts[pos]
        # Count alphabetic letters
        alpha_len = sum(1 for c in word if c.isalpha())
        if alpha_len >= min_word_len and rng.random() < mask_prob:
            masked.add(pos)
            # Span masking: with span_prob extend to next 1-3 tokens
            if rng.random() < span_prob:
                span_len = rng.randint(1, 3)
                for j in range(1, span_len + 1):
                    if i + j < len(token_positions):
                        masked.add(token_positions[i + j])
                i += span_len  # skip over the spanned tokens
        i += 1

    result_parts = []
    for i, p in enumerate(parts):
        if i in masked:
            result_parts.append("[MASK]")
        else:
            result_parts.append(p)
    return "".join(result_parts)


# ---------------------------------------------------------------------------
# Prompt building (mirrors _build_generator_generate_prompt)
# ---------------------------------------------------------------------------


def build_messages(query: str, retriever_message: str) -> list[dict]:
    """Build the messages list for the generator model."""
    user_parts = [f"Research query:\n{query}"]
    user_parts.append(f"\n[Initial retriever synthesis]\n{retriever_message}")
    user_content = "\n".join(user_parts)
    return [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> tuple[str, str]:
    """Return (reasoning, content) from the model."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.95,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
    )
    choice = response.choices[0].message
    reasoning = getattr(choice, "reasoning", "") or ""
    content = choice.content or ""
    return reasoning.strip(), content.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hypothesis generator inference over retriever outputs."
    )
    parser.add_argument(
        "--retriever-out",
        type=Path,
        default=ROOT / "data/rl_retriever_outputs.csv",
        help="Input CSV produced by run_retriever_inference.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/rl_generator_outputs.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible base URL (default: http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name as registered in vLLM (default: Qwen/Qwen3-32B).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of parallel inference threads (default: 64).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Maximum number of rows to process (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for masking.",
    )
    parser.add_argument(
        "--use-masking",
        action="store_true",
        default=False,
        help="Apply span masking to the retriever message before prompting the generator.",
    )
    parser.add_argument(
        "--mask-prob",
        type=float,
        default=0.15,
        help="Probability of masking each eligible word (default: 0.15).",
    )
    parser.add_argument(
        "--span-prob",
        type=float,
        default=0.2,
        help="Probability of extending a mask to a span of 1-3 additional tokens (default: 0.2).",
    )
    parser.add_argument(
        "--min-word-len",
        type=int,
        default=3,
        help="Minimum number of alphabetic characters for a word to be eligible for masking (default: 3).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    client = OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=3600)

    print(f"Loading retriever outputs from {args.retriever_out} …")
    rows = []
    with open(args.retriever_out, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"  {len(rows)} rows loaded.")

    if args.n is not None:
        rows = rows[: args.n]
        print(f"  Limiting to {len(rows)} rows (--n {args.n}).")

    if not rows:
        print("No rows found. Exiting.")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "paper_id",
        "paper_title",
        "user_query",
        "retriever_message",
        "masked_retriever_message",
        "input_messages",
        "reasoning",
        "hypotheses",
    ]

    write_lock = threading.Lock()
    completed = 0
    total = len(rows)

    def process(row: dict, index: int) -> bool:
        paper_id = row.get("paper_id", "")
        paper_title = row.get("paper_title", "")
        user_query = row.get("user_query", "").strip()
        retriever_message = row.get("retriever_message", "").strip()

        if args.use_masking:
            masked_message = apply_span_masking(
                retriever_message,
                rng,
                mask_prob=args.mask_prob,
                span_prob=args.span_prob,
                min_word_len=args.min_word_len,
            )
        else:
            masked_message = retriever_message

        messages = build_messages(user_query, masked_message)

        print(f"\n[{index}/{total}] {paper_title[:70]!r}")

        try:
            reasoning, content = call_model(
                client, args.model, messages, args.max_tokens
            )
            print(
                f"  reasoning: {len(reasoning)} chars | hypotheses: {len(content)} chars"
            )
        except Exception as exc:
            print(f"  ERROR: {exc} — skipping.")
            return False

        out_row = {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "user_query": user_query,
            "retriever_message": retriever_message,
            "masked_retriever_message": masked_message if args.use_masking else "",
            "input_messages": json.dumps(messages, ensure_ascii=False),
            "reasoning": reasoning,
            "hypotheses": content,
        }
        with write_lock:
            writer.writerow(out_row)
            f.flush()
        return True

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process, row, i): row for i, row in enumerate(rows, 1)
            }
            for future in as_completed(futures):
                if future.result():
                    completed += 1

    print(f"\nDone. {completed}/{total} samples saved to {args.out}")


if __name__ == "__main__":
    main()
