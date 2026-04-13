import os
import sys
import json
import asyncio
import csv
import random
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.sft.utils.config import CONFIG

load_dotenv(os.path.join(AGENTS_DIR, ".env"))

DATASETS_DIR = os.path.join(AGENTS_DIR, CONFIG["paths"].get("datasets", "data/datasets"))

INPUT_FILE_NAME = CONFIG["files"].get("synthetic_sft_dataset", "sft_dataset_v2.jsonl")
INPUT_JSONL = os.path.join(DATASETS_DIR, INPUT_FILE_NAME)
OUTPUT_FILE_NAME = CONFIG["files"].get("updated_sft_dataset", "multiagent_sft_dataset_v2.jsonl")
OUTPUT_JSONL = os.path.join(DATASETS_DIR, OUTPUT_FILE_NAME)
OUTPUT_CSV = OUTPUT_JSONL.replace(".jsonl", ".csv")

print("Weryfikacja sczek:")
print(f"AGENTS_DIR: {AGENTS_DIR}")
print(f"DATASETS_DIR: {DATASETS_DIR}")
print(f"INPUT_JSONL: {INPUT_JSONL}")
print(f"OUTPUT_JSONL: {OUTPUT_JSONL}")
print(f"OUTPUT_CSV: {OUTPUT_CSV}")
print("----------------------------------------")

SEMAPHORE_SIZE = 5
TEMPERATURE = 0.25
MODEL_NAME = "gemini-3-flash-preview"

llm_retriever = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_retries=2,
    model_kwargs={"reasoning_effort": "low"}
)

llm_generator = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_retries=2,
    model_kwargs={"reasoning_effort": "medium"}
)

class RetrieverMessageStr(BaseModel):
    natural_message: str

class GeneratorResponseStr(BaseModel):
    thought_process: str
    final_output: str

llm_retriever_formatter = llm_retriever.with_structured_output(RetrieverMessageStr)
llm_generator_formatter = llm_generator.with_structured_output(GeneratorResponseStr)

prompt_format_retriever = ChatPromptTemplate.from_messages([
    ("system", """You are the Retriever agent in a scientific multi-agent system.
Your task is to transfer condensed knowledge to the Generator agent.
Based on the provided JSON data (variables, relationships, evidence), write a concise, professional, natural-language message summarizing what you found. 
Do NOT hallucinate or add any information outside of the provided JSON."""),
    ("human", """USER QUERY: 
{query}

EXTRACTED DATA (JSON): 
{extracted_info}""")
])

prompt_format_generator_success = ChatPromptTemplate.from_messages([
    ("system", """You are the Generator agent. You received a message from the Retriever and successfully formed a hypothesis.
Populate the required structured output fields as follows:
- 'thought_process': Write 1-2 sentences of internal reasoning stating that the context is sufficient to form a causal hypothesis.
- 'final_output': EXACTLY copy the provided HYPOTHESIS. Do not alter it.
Do NOT output markdown blocks or numbered lists."""),
    ("human", """RETRIEVER MESSAGE: 
{retriever_msg}

YOUR ORIGINAL REASONING: 
{reasoning}

HYPOTHESIS TO COPY: 
{hypothesis}""")
])

prompt_format_generator_fail = ChatPromptTemplate.from_messages([
    ("system", """You are the Generator agent. You received a message from the Retriever, but the data is INSUFFICIENT to build a rigorous hypothesis.
Populate the required structured output fields as follows:
- 'thought_process': Write 1-2 sentences of internal reasoning explaining what critical information is missing based on the provided reasoning.
- 'final_output': Write a direct request to the Retriever asking for specific missing information from the articles. You MUST wrap your request in a <REQUEST>...</REQUEST> tag.
Do NOT output markdown blocks or numbered lists."""),
    ("human", """RETRIEVER MESSAGE: 
{retriever_msg}

WHY DATA WAS INSUFFICIENT (REASONING): 
{reasoning}""")
])

write_lock = asyncio.Lock()

def load_existing_prompt_ids(filepath: str) -> set:
    if not os.path.exists(filepath):
        return set()
    existing_ids = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    existing_ids.add(record["prompt_id"])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Could not read existing IDs from {filepath}: {e}")
    return existing_ids

async def process_record(record, semaphore, f_out, csv_writer, f_csv):
    async with semaphore:
        prompt_id = record.get("prompt_id")

        query = record.get("user_query")
        extracted_info = record.get("retriever_extracted_info")
        is_answerable = record.get("generator_is_answerable")
        reasoning = record.get("generator_reasoning")
        hypothesis = record.get("generator_hypothesis")

        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                # RETRIEVER
                retriever_result = await (prompt_format_retriever | llm_retriever_formatter).ainvoke({
                    "query": query,
                    "extracted_info": extracted_info
                })

                retriever_msg = retriever_result.natural_message

                # GENERATOR
                if is_answerable:
                    gen_result = await (prompt_format_generator_success | llm_generator_formatter).ainvoke({
                        "retriever_msg": retriever_msg,
                        "reasoning": reasoning,
                        "hypothesis": hypothesis
                    })
                else:
                    gen_result = await (prompt_format_generator_fail | llm_generator_formatter).ainvoke({
                        "retriever_msg": retriever_msg,
                        "reasoning": reasoning
                    })

                generator_full = f"<THOUGHT>\n{gen_result.thought_process}\n</THOUGHT>\n{gen_result.final_output}"

                conversation = [
                    {"role": "user", "content": query},
                    {"role": "retriever", "content": retriever_msg},
                    {"role": "generator", "content": generator_full}
                ]

                new_record = {
                    "prompt_id": prompt_id,
                    "is_success": is_answerable,
                    "user_query": query,
                    "raw_context": record.get("raw_context"),
                    "retriever_extracted_info": extracted_info,
                    "retriever_message": retriever_msg,
                    "generator_reasoning": reasoning,
                    "generator_response": generator_full,
                    "messages": conversation
                }

                csv_record = {
                    "prompt_id": prompt_id,
                    "is_success": is_answerable,
                    "user_query": query,
                    "raw_context": record.get("raw_context"),
                    "retriever_extracted_info": extracted_info,
                    "retriever_message": retriever_msg,
                    "generator_reasoning": reasoning,
                    "generator_response": generator_full
                }

                async with write_lock:
                    f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                    f_out.flush()

                    csv_writer.writerow(csv_record)
                    f_csv.flush()

                return True

            except Exception as e:
                err = str(e)

                if any(x in err for x in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500"]):
                    wait = min(120, 10 * (2 ** attempt)) + random.uniform(0, 5) 
                    code = "503" if "503" in err else "429"
                    tqdm.write(f"[{code} API BUSY] id={prompt_id} wait={wait:.1f}s attempt={attempt+1}")
                    await asyncio.sleep(wait)

                elif "Invalid json output" in err or "OutputParserException" in err:
                    tqdm.write(f"[PARSER ERR] id={prompt_id} zły format JSON, ponawiam próbę {attempt+1}/{max_attempts}")
                    continue 
                    
                else:
                    tqdm.write(f"[FATAL ERR {prompt_id}] {err}")
                    return False

        tqdm.write(f"[FAILED] {prompt_id}")
        return False


async def main():
    if not os.path.exists(INPUT_JSONL):
        print("Brak pliku wejściowego.")
        return

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"Wczytano rekordów: {len(records)}")

    existing_ids = load_existing_prompt_ids(OUTPUT_JSONL)
    if existing_ids:
        print(f"Checkpoint: pominięto {len(existing_ids)} już przetworzonych rekordów.")
        records = [r for r in records if r.get("prompt_id") not in existing_ids]

    print(f"Do przetworzenia: {len(records)}")

    if not records:
        print("Wszystkie rekordy już przetworzone.")
        return

    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)
    success = 0

    with (
        open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out,
        open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f_csv
    ):
        writer = csv.DictWriter(f_csv, fieldnames=[
            "prompt_id", "is_success", "user_query", 
            "raw_context", "retriever_extracted_info", 
            "retriever_message", "generator_reasoning", "generator_response"
        ])

        if f_csv.tell() == 0:
            writer.writeheader()

        tasks = [
            process_record(rec, semaphore, f_out, writer, f_csv)
            for rec in records
        ]

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            if await future:
                success += 1

    print(f"\nDONE: {success}/{len(records)}")


if __name__ == "__main__":
    asyncio.run(main())