"""OpenAI-compatible API client.

Works with any endpoint that speaks the OpenAI Chat Completions API:

    # OpenAI
    client = OpenAICompatibleClient(model="gpt-4o-mini")

    # vLLM (serve with: vllm serve Qwen/Qwen2.5-7B-Instruct)
    client = OpenAICompatibleClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key="dummy",
    )

    # Ollama
    client = OpenAICompatibleClient(
        model="llama3",
        base_url="http://localhost:11434/v1",
    )

Environment variable overrides (used when constructor args are None):
    OPENAI_API_KEY   — authentication key (use "dummy" for local servers)
    OPENAI_BASE_URL  — base URL of the endpoint (default: OpenAI)
    OPENAI_MODEL     — model name fallback
"""

import json
import os

from app.api_client.base import BaseAPIClient, CallResult, Message


class OpenAICompatibleClient(BaseAPIClient):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI  # noqa: PLC0415

        self.model: str = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def call(self, messages: list[Message], response_schema=None) -> CallResult:
        oai_messages = [{"role": m.role, "content": m.content} for m in messages]

        if response_schema is not None:
            try:
                # Structured output — supported by OpenAI and some vLLM deployments
                response = self._client.beta.chat.completions.parse(
                    model=self.model,
                    messages=oai_messages,
                    response_format=response_schema,
                )
                return CallResult(
                    content=response.choices[0].message.parsed,
                    model=self.model,
                )
            except Exception:
                # Fallback: JSON mode + manual Pydantic parse
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=oai_messages,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or "{}"
                data = json.loads(raw)
                return CallResult(content=response_schema(**data), model=self.model)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
        )
        return CallResult(
            content=response.choices[0].message.content or "",
            model=self.model,
        )
