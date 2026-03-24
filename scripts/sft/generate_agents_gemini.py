import os
import sys
import asyncio
import pandas as pd
import csv
import json
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DISK_DIR = os.path.dirname(AGENTS_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.sft.utils.config import CONFIG

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(os.path.join(AGENTS_DIR, ".env"))

LOCAL_BASE = os.path.abspath(CONFIG["paths"].get("base_path_local", "./"))

PROMPT_EMBEDDINGS_FILE = os.path.join(AGENTS_DIR, "data", "embeddings", CONFIG["files"]["prompt_embeddings"])
RETRIEVED_CONTEXTS_FILE = os.path.join(AGENTS_DIR, "data", "retrieval", CONFIG["files"]["retrieved_contexts"])

OUTPUT_BASE = os.path.join(AGENTS_DIR, "data", "datasets", CONFIG["files"]["synthetic_sft_dataset"])
OUTPUT_FILE_JSONL = OUTPUT_BASE if OUTPUT_BASE.endswith(".jsonl") else OUTPUT_BASE + ".jsonl"
OUTPUT_FILE_CSV = OUTPUT_BASE.replace(".jsonl", ".csv") if OUTPUT_BASE.endswith(".jsonl") else OUTPUT_BASE + ".csv"

os.makedirs(os.path.dirname(OUTPUT_FILE_JSONL), exist_ok=True)

SEMAPHORE_SIZE = CONFIG["inference"].get("semaphore_size", 5)
GEN_MODEL = CONFIG["models"].get("generation_model")
TEMPERATURE = CONFIG["inference"].get("temperature", 0.4)

# list all paths to verify
print(f"Prompt Embeddings Path: {PROMPT_EMBEDDINGS_FILE}")
print(f"Retrieved Contexts Path: {RETRIEVED_CONTEXTS_FILE}")
print(f"Output JSONL Path: {OUTPUT_FILE_JSONL}")
print(f"Output CSV Path: {OUTPUT_FILE_CSV}")


class ExtractedInformation(BaseModel):
    variables: List[str]
    relationships: List[str]
    mechanisms: List[str]
    evidence: List[str]

class RetrieverMessage(BaseModel):
    is_sufficient: bool
    reasoning: str
    extracted_information: ExtractedInformation

class GeneratorHypothesis(BaseModel):
    is_answerable: bool
    reasoning: str
    hypothesis_statement: str
    natural_hypothesis: str
    falsification_criteria: str


llm_retriever = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", temperature=0.1, max_retries=5
).with_structured_output(RetrieverMessage)

llm_generator = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview", temperature=TEMPERATURE, max_retries=5
).with_structured_output(GeneratorHypothesis)

retriever_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an Expert Scientific Retriever Agent.

Return:

1. is_sufficient:
- True if enough info for hypothesis
- False otherwise

2. reasoning:
- Step-by-step explanation of relevance

3. extracted_information:
- variables: list
- relationships: list
- mechanisms: list
- evidence: list

IMPORTANT:
- Do NOT generate hypotheses
- Be strict: mark insufficient if unsure""",
        ),
        (
            "human",
            """RESEARCH QUERY:
{query}

RAW CONTEXT:
{context}""",
        ),
    ]
)

generator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI Research Scientist generating hypotheses.

You will receive:
- Research Query
- Raw Context
- Retriever Reasoning
- Extracted Information
- Sufficiency flag

Return:

1. reasoning:
- Explain how info leads to hypothesis

2. hypothesis_statement (STRUCTURED):
- MUST follow:
  "If [IV] changes, then [DV] will [effect], because [mechanism]."

3. natural_hypothesis:
- Same hypothesis written in natural academic style

4. falsification_criteria:
- Specific measurable condition that disproves hypothesis

IMPORTANT:
- Use ONLY variables from extracted information
- Do NOT invent new variables
- If insufficient → is_answerable = False
- If is_sufficient = False, you MUST set is_answerable = False.""",
        ),
        (
            "human",
            """RESEARCH QUERY:
{query}

RAW CONTEXT:
{raw_context}

RETRIEVER SUFFICIENCY:
{is_sufficient}

RETRIEVER REASONING:
{retriever_reasoning}

EXTRACTED INFORMATION:
{retrieved_info}""",
        ),
    ]
)

write_lock = asyncio.Lock()


async def process_item(row, semaphore: asyncio.Semaphore, csv_writer, f_csv, f_jsonl):
    async with semaphore:
        prompt_id = row["prompt_id"]
        query_text = row["text"]
        raw_context = row["retrieved_context"]

        try:
            retriever_inputs = {"query": query_text, "context": raw_context}

            retriever_output = await (retriever_prompt | llm_retriever).ainvoke(
                retriever_inputs
            )

            if len(retriever_output.extracted_information.variables) == 0:
                retriever_output.is_sufficient = False

            generator_inputs = {
                "query": query_text,
                "raw_context": raw_context,
                "is_sufficient": retriever_output.is_sufficient,
                "retriever_reasoning": retriever_output.reasoning,
                "retrieved_info": retriever_output.extracted_information.model_dump(),
            }

            generator_output = await (generator_prompt | llm_generator).ainvoke(
                generator_inputs
            )

            record = {
                "prompt_id": int(prompt_id),
                "user_query": query_text,
                "raw_context": raw_context,
                "retriever_is_sufficient": retriever_output.is_sufficient,
                "retriever_reasoning": retriever_output.reasoning,
                "retriever_extracted_info": json.dumps(
                    retriever_output.extracted_information.model_dump(),
                    ensure_ascii=False
                ),
                "generator_is_answerable": generator_output.is_answerable,
                "generator_reasoning": generator_output.reasoning,
                "generator_hypothesis": generator_output.hypothesis_statement,
                "generator_natural_hypothesis": generator_output.natural_hypothesis,
                "generator_falsification": generator_output.falsification_criteria,
            }

            async with write_lock:
                csv_writer.writerow(record)
                f_csv.flush()

                f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_jsonl.flush()

            return True

        except Exception as e:
            print(f"\n[Error on prompt {prompt_id}]: {str(e)}")
            return False


async def main():
    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: Before running the script, set the API key")
        return

    print("Loading data...")
    try:
        df_prompts = pd.read_pickle(PROMPT_EMBEDDINGS_FILE)
        df_contexts = pd.read_pickle(RETRIEVED_CONTEXTS_FILE)

        df_merged = pd.merge(
            df_prompts[["prompt_id", "text"]], df_contexts, on="prompt_id"
        )
        print(f"Merged: {len(df_merged)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)
    success_count = 0

    print(f"Generating SFT dataset → JSONL & CSV")

    fieldnames = [
        "prompt_id",
        "user_query",
        "raw_context",
        "retriever_is_sufficient",
        "retriever_reasoning",
        "retriever_extracted_info",
        "generator_is_answerable",
        "generator_reasoning",
        "generator_hypothesis",
        "generator_natural_hypothesis",
        "generator_falsification",
    ]

    file_exists_csv = (
        os.path.isfile(OUTPUT_FILE_CSV) and os.path.getsize(OUTPUT_FILE_CSV) > 0
    )

    with (
        open(OUTPUT_FILE_JSONL, "a", encoding="utf-8") as f_jsonl,
        open(OUTPUT_FILE_CSV, "a", newline="", encoding="utf-8") as f_csv,
    ):
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)

        if not file_exists_csv:
            writer.writeheader()

        tasks = [
            process_item(row, semaphore, writer, f_csv, f_jsonl)
            for _, row in df_merged.iterrows()
        ]

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            if await future:
                success_count += 1

    print("\n" + "=" * 50)
    print(f"Generated: {success_count}/{len(tasks)} records")
    print(f"Saved JSONL to: {OUTPUT_FILE_JSONL}")
    print(f"Saved CSV to: {OUTPUT_FILE_CSV}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())