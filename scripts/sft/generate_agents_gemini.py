import os
import sys
import asyncio
import pandas as pd
import csv
import json
from tqdm import tqdm
from pydantic import BaseModel, Field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

from sft.utils.config import CONFIG

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(os.path.join(AGENTS_DIR, ".env"))

LOCAL_BASE = os.path.abspath(CONFIG["paths"].get("base_path_local", "./"))

PROMPT_EMBEDDINGS_FILE = os.path.join(LOCAL_BASE, CONFIG["files"]["prompt_embeddings"])
RETRIEVED_CONTEXTS_FILE = os.path.join(
    LOCAL_BASE, CONFIG["files"]["retrieved_contexts"]
)
OUTPUT_BASE = os.path.join(LOCAL_BASE, CONFIG["files"]["synthetic_sft_dataset"])
OUTPUT_FILE_JSONL = (
    OUTPUT_BASE if OUTPUT_BASE.endswith(".jsonl") else OUTPUT_BASE + ".jsonl"
)
OUTPUT_FILE_CSV = (
    OUTPUT_BASE.replace(".jsonl", ".csv")
    if OUTPUT_BASE.endswith(".jsonl")
    else OUTPUT_BASE + ".csv"
)

os.makedirs(os.path.dirname(OUTPUT_FILE_JSONL), exist_ok=True)

SEMAPHORE_SIZE = CONFIG["inference"].get("semaphore_size", 5)
GEN_MODEL = CONFIG["models"].get("generation_model", "gemini-1.5-pro")
TEMPERATURE = CONFIG["inference"].get("temperature", 0.4)

# list all paths to verify
print(f"Prompt Embeddings Path: {PROMPT_EMBEDDINGS_FILE}")
print(f"Retrieved Contexts Path: {RETRIEVED_CONTEXTS_FILE}")
print(f"Output JSONL Path: {OUTPUT_FILE_JSONL}")
print(f"Output CSV Path: {OUTPUT_FILE_CSV}")


class RetrieverMessage(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning explaining how the context relates to the query. Identify relevant vs irrelevant parts and explain mechanisms."
    )
    extracted_information: str = Field(
        description="Structured extraction of key variables, relationships, mechanisms, and data points relevant to the query."
    )


class GeneratorHypothesis(BaseModel):
    is_answerable: bool = Field(
        description="True ONLY if the retriever output contains enough grounded information."
    )
    reasoning: str = Field(
        description="Scientific reasoning explaining how extracted information leads to the hypothesis."
    )
    hypothesis_statement: str = Field(
        description="MUST follow: 'If [Independent Variable] changes, then [Dependent Variable] will [direction], because [Mechanism].'"
    )
    falsification_criteria: str = Field(
        description="A specific, measurable experimental result that would prove the hypothesis WRONG."
    )


llm_retriever = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.1, max_retries=5
).with_structured_output(RetrieverMessage)

llm_generator = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", temperature=TEMPERATURE, max_retries=5
).with_structured_output(GeneratorHypothesis)

retriever_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an Expert Scientific Retriever Agent.

Your job is to analyze raw scientific context and prepare structured knowledge for hypothesis generation.

Your output MUST contain:

1. Reasoning:
- Step-by-step explanation of how the context relates to the query
- Identify relevant vs irrelevant parts
- Explain mechanisms and relationships

2. Extracted Information:
- Key variables
- Relationships
- Mechanisms
- Data points (if available)

IMPORTANT:
- Do NOT generate hypotheses
- If context is insufficient, clearly state it in reasoning and extracted information""",
        ),
        (
            "human",
            """**RESEARCH QUERY:**
{query}

**RAW RETRIEVED CONTEXT:**
{context}""",
        ),
    ]
)

generator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI Research Scientist generating scientific hypotheses.

You will receive:
- Research Query
- Retriever Reasoning
- Extracted Information

Your output MUST contain:

1. Reasoning:
- Step-by-step scientific reasoning
- Explain how extracted information leads to the hypothesis

2. Hypothesis:
- Write it in a natural, highly professional academic style.
- Clearly state the proposed relationships, effects, or mechanisms.
- Do NOT use rigid school templates (like "If... then..."). Write like a PhD researcher.

3. Falsification criteria:
- A specific measurable condition that would disprove the hypothesis.

IMPORTANT:
- Hypothesis must be grounded in extracted information.
- Do NOT introduce new variables.
- If information is insufficient, set is_answerable = False.""",
        ),
        (
            "human",
            """**RESEARCH QUERY:**
{query}

**RETRIEVER REASONING:**
{retriever_reasoning}

**EXTRACTED INFORMATION:**
{retrieved_info}""",
        ),
    ]
)


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

            generator_inputs = {
                "query": query_text,
                "retriever_reasoning": retriever_output.reasoning,
                "retrieved_info": retriever_output.extracted_information,
            }
            generator_output = await (generator_prompt | llm_generator).ainvoke(
                generator_inputs
            )

            record = {
                "prompt_id": int(prompt_id),
                "user_query": query_text,
                "raw_context": raw_context,
                "retriever_reasoning": retriever_output.reasoning,
                "retriever_extracted_info": retriever_output.extracted_information,
                "generator_is_answerable": generator_output.is_answerable,
                "generator_reasoning": generator_output.reasoning,
                "generator_hypothesis": generator_output.hypothesis_statement,
                "generator_falsification": generator_output.falsification_criteria,
            }

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
        "retriever_reasoning",
        "retriever_extracted_info",
        "generator_is_answerable",
        "generator_reasoning",
        "generator_hypothesis",
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
