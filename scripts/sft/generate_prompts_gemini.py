import os
import sys
import json
import csv
import asyncio
import random
from tqdm import tqdm
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.sft.utils.config import CONFIG

from dotenv import load_dotenv

env_path = os.path.join(AGENTS_DIR, ".env")
load_dotenv(env_path)
RAW_BASE_PATH = CONFIG["paths"]["base_path_local"]
ABS_BASE_PATH = os.path.expanduser(RAW_BASE_PATH)

PROMPTS_DIR = os.path.join(AGENTS_DIR, "data", "prompts")
TOPICS_FILE = os.path.join(PROMPTS_DIR, CONFIG["files"]["prompt_topics"])
PROMPTS_FILE = os.path.join(PROMPTS_DIR, CONFIG["files"]["prompts"])

os.makedirs(PROMPTS_DIR, exist_ok=True)

TEMPERATURE = CONFIG["inference"].get("temperature", 0.7)
SEMAPHORE_SIZE = CONFIG["inference"].get("semaphore_size", 10)
TARGET_PROMPT_COUNT = CONFIG["inference"].get("target_prompt_count", 500)
GEN_MODEL = CONFIG["models"].get("generation_model", "gemini-1.5-pro")


class SyntheticUserPrompt(BaseModel):
    generated_prompt_text: str = Field(
        description="The realistic, generated research query (prompt) that a user would type."
    )
    intent_category: str = Field(
        description="The intent category of the query, e.g., 'Method Comparison', 'Troubleshooting', 'Theoretical Inquiry', 'Performance Optimization'."
    )


llm = ChatGoogleGenerativeAI(
    model=GEN_MODEL, temperature=TEMPERATURE, max_retries=3
).with_structured_output(SyntheticUserPrompt)

system_template = """A 
You are an expert user simulator for a scientific AI system.
Your goal is to generate a realistic, prompt that a professional AI scientist or researcher would type into a system used to generate scientific hypotheses.

**ROLEPLAYING INSTRUCTIONS:**
1. Analyze the provided Scientific Topic, its Discipline, and its Subtopics.
2. Generate ONE specific research query.
3. The query MUST NOT be trivial. It should explore deep relationships, optimizations, or open research problems.
4. Use advanced scientific vocabulary, mathematical terms, and engineering jargon appropriate for a PhD-level researcher.
5. FORMATTING: Make it sound like a natural, direct question a busy researcher would ask an AI assistant. DO NOT use essay-like openers like "Elucidate...", "Critically compare...", or "Considering...". Keep it concise, practical, and direct (max 1-2 sentences).
"""

human_template = """
**SCIENTIFIC TOPIC:**
Discipline: {discipline}
Topic Name: {topic_name}
Description: {topic_description}
Key Subtopics: {topic_subtopics}

Generate the user prompt now.
"""

prompt_chain = ChatPromptTemplate.from_messages(
    [("system", system_template), ("human", human_template)]
)


def load_topics(topics_path: str) -> list:
    if not os.path.exists(topics_path):
        raise FileNotFoundError(f"Topic file not found: {topics_path}")

    topics = []
    with open(topics_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    if "subjects" in record and isinstance(record["subjects"], list):
                        for subject in record["subjects"]:
                            subject["parent_discipline"] = record.get(
                                "discipline", "Unknown"
                            )
                            topics.append(subject)
                    elif "name" in record:
                        topics.append(record)
                except json.JSONDecodeError:
                    continue
    return topics


async def process_item(topic: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        inputs = {
            "discipline": topic.get("parent_discipline", "Unknown AI Field"),
            "topic_name": topic.get("name", ""),
            "topic_description": topic.get("description", ""),
            "topic_subtopics": ", ".join(topic.get("key_subtopics", [])),
        }

        try:
            response_data = await (prompt_chain | llm).ainvoke(inputs)

            return {
                "status": "success",
                "topic_id": topic.get("id", topic.get("name")),
                "topic_name": topic.get("name"),
                "generated_prompt": response_data.generated_prompt_text,
                "intent": response_data.intent_category,
            }
        except Exception as e:
            return {"status": "error", "topic_id": topic.get("name"), "error": str(e)}


async def main():
    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: Before running the script, set the API key in the console!")
        print('Type: $env:GOOGLE_API_KEY="YOUR_API_KEY"')
        return

    print(f"Looking for topic file in: {TOPICS_FILE}")
    try:
        all_topics = load_topics(TOPICS_FILE)
    except FileNotFoundError as e:
        print(e)
        return

    # drawing topics
    if len(all_topics) > TARGET_PROMPT_COUNT:
        print(
            f"Found {len(all_topics)} topics. Sampling {TARGET_PROMPT_COUNT} for generation..."
        )
        topics_to_run = random.sample(all_topics, TARGET_PROMPT_COUNT)
    else:
        print(f"Found {len(all_topics)} topics. Processing all...")
        topics_to_run = all_topics

    # prepare CSV file for output
    CSV_FIELDNAMES = ["topic_id", "topic_name", "generated_prompt", "intent"]
    file_exists = os.path.exists(PROMPTS_FILE)

    csv_file = open(PROMPTS_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)

    if not file_exists or os.path.getsize(PROMPTS_FILE) == 0:
        writer.writeheader()
        csv_file.flush()

    # prepare asynchronous configuration
    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)
    tasks = [process_item(topic, semaphore) for topic in topics_to_run]

    success_count = 0
    fail_count = 0

    print(f"Starting asynchronous calls to Gemini API for {len(tasks)} topics...")

    try:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await future

            if result["status"] == "success":
                writer.writerow(
                    {
                        "topic_id": result["topic_id"],
                        "topic_name": result["topic_name"],
                        "generated_prompt": result["generated_prompt"],
                        "intent": result["intent"],
                    }
                )
                csv_file.flush()
                success_count += 1
            else:
                print(
                    f"\nError for topic '{result.get('topic_id')}' — {result.get('error')}"
                )
                fail_count += 1
    finally:
        csv_file.close()

    print("-" * 30)
    print("Summary:")
    print(f"  Total Attempts : {len(tasks)}")
    print(f"  Successes      : {success_count}")
    print(f"  Errors         : {fail_count}")
    print("-" * 30)


if __name__ == "__main__":
    asyncio.run(main())
