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
            """You are an Expert Scientific Information Extractor.

RULES:
- Extract ONLY explicitly stated information
- Do NOT infer beyond text
- Do NOT invent variables

Return:

1. is_sufficient:
True ONLY if:
- ≥2 variables
- ≥1 relationship

2. reasoning:
Max 3 sentences

3. extracted_information:
- variables
- relationships
- mechanisms
- evidence

Be strict.""",
        ),
        (
            "human",
            """QUERY:
{query}

CONTEXT:
{context}""",
        ),
    ]
)

generator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI Research Scientist.

Generate ONE high-quality hypothesis.

RULES:
- Must be causal
- Must be measurable
- Must be grounded in context
- Use ONLY variables listed in EXTRACTED. Do not introduce new variables. If variables are unclear → set is_answerable = False.

FORMAT STRICTLY:

"If X increases/decreases, then Y will [effect], because [mechanism]."

STRICT:
- No placeholders
- No brackets
- No "Not applicable"
- No vague statements

Return:
- is_answerable
- reasoning
- hypothesis_statement
- natural_hypothesis
- falsification_criteria""",
        ),
        (
            "human",
            """QUERY:
{query}

CONTEXT:
{raw_context}

SUFFICIENCY:
{is_sufficient}

EXTRACTED:
{retrieved_info}""",
        ),
    ]
)

def validate_retriever_output(output):
    vars_ok = len(output.extracted_information.variables) >= 2
    rel_ok = len(output.extracted_information.relationships) >= 1

    if not (vars_ok and rel_ok):
        output.is_sufficient = False

    return output


def clean_extracted_info(info):
    return {
        "variables": list(set(info.variables))[:5],
        "relationships": info.relationships[:3],
        "mechanisms": info.mechanisms[:2],
    }


def is_valid_hypothesis(output):
    h = output.hypothesis_statement

    if h is None:
        return False

    banned = ["If [IV]", "[DV]", "Not applicable", "N/A"]

    if any(b in h for b in banned):
        return False

    if len(h) < 30:
        return False

    return True

write_lock = asyncio.Lock()


async def process_item(row, semaphore, csv_writer, f_csv, f_jsonl):
    async with semaphore:
        prompt_id = row["prompt_id"]
        query_text = row["text"]
        raw_context = row["retrieved_context"]

        try:
            retriever_output = await (retriever_prompt | llm_retriever).ainvoke(
                {"query": query_text, "context": raw_context}
            )

            retriever_output = validate_retriever_output(retriever_output)

            if not retriever_output.is_sufficient:
                return False 

            cleaned_info = clean_extracted_info(
                retriever_output.extracted_information
            )


            generator_output = await (generator_prompt | llm_generator).ainvoke(
                {
                    "query": query_text,
                    "raw_context": raw_context,
                    "is_sufficient": retriever_output.is_sufficient,
                    "retrieved_info": cleaned_info,
                }
            )

            if not generator_output.is_answerable:
                return False

            if not is_valid_hypothesis(generator_output):
                return False

            record = {
                "prompt_id": int(prompt_id),
                "user_query": query_text,
                "raw_context": raw_context,
                "retriever_is_sufficient": retriever_output.is_sufficient,
                "retriever_reasoning": retriever_output.reasoning,
                "retriever_extracted_info": json.dumps(cleaned_info, ensure_ascii=False),
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
        print("ERROR: Set API key")
        return

    df_prompts = pd.read_pickle(PROMPT_EMBEDDINGS_FILE)
    df_contexts = pd.read_pickle(RETRIEVED_CONTEXTS_FILE)

    df_merged = pd.merge(
        df_prompts[["prompt_id", "text"]],
        df_contexts,
        on="prompt_id",
    )

    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)

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

    with (
        open(OUTPUT_FILE_JSONL, "a", encoding="utf-8") as f_jsonl,
        open(OUTPUT_FILE_CSV, "a", newline="", encoding="utf-8") as f_csv,
    ):
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)

        if f_csv.tell() == 0:
            writer.writeheader()

        tasks = [
            process_item(row, semaphore, writer, f_csv, f_jsonl)
            for _, row in df_merged.head(100).iterrows()
        ]

        success = 0

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            if await future:
                success += 1

    print(f"\nGenerated: {success}/{len(tasks)}")


if __name__ == "__main__":
    asyncio.run(main())