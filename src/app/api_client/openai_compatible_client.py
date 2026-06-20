"""OpenAI-compatible API client.

Works with any endpoint that speaks the OpenAI Chat Completions API:

    # OpenAI
    client = OpenAICompatibleClient(model="gpt-4o-mini")

    # vLLM (serve with: ./sh_scripts/serve_vllm.sh)
    client = OpenAICompatibleClient(
        model="/models/example_checkpoint/global_step75_hf",
        base_url="http://localhost:8017/v1",
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

Reasoning models (e.g. Qwen3 served by vLLM with --reasoning-parser qwen3):
    If the server populates ``reasoning_content`` on the response message the
    text is captured in ``CallResult.reasoning``.  If the model instead emits
    ``<think>…</think>`` inline, the block is stripped from ``content`` and
    the extracted text is placed in ``CallResult.reasoning``.  For providers
    that do neither, ``reasoning`` is simply an empty string.
"""

import json
import os
import re

from app.api_client.base import BaseAPIClient, CallResult, Message

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_reasoning(text: str) -> tuple[str, str]:
    """Strip ``<think>…</think>`` blocks from *text*.

    Returns ``(reasoning, answer)`` — reasoning is the concatenated content of
    all think blocks; answer is the remaining text.  Both are stripped of
    leading/trailing whitespace.
    """
    parts: list[str] = []

    def _collect(m: re.Match) -> str:
        parts.append(m.group(1).strip())
        return ""

    answer = _THINK_RE.sub(_collect, text).strip()
    return "\n\n".join(parts), answer


def _reasoning_content(choice) -> str:
    """Return ``reasoning_content`` set by vLLM's reasoning parser, or ``""``."""
    return getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None) or ""


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
            return self._call_structured(oai_messages, response_schema)
        return self._call_plain(oai_messages)

    def _call_plain(self, oai_messages: list[dict]) -> CallResult:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
        )
        choice = response.choices[0]
        reasoning = _reasoning_content(choice)
        raw = choice.message.content or ""
        if not reasoning:
            reasoning, raw = _split_reasoning(raw)
        return CallResult(content=raw, model=self.model, reasoning=reasoning)

    def _call_structured(self, oai_messages: list[dict], response_schema) -> CallResult:
        try:
            # Structured output — supported by OpenAI and some vLLM deployments
            response = self._client.beta.chat.completions.parse(
                model=self.model,
                messages=oai_messages,
                response_format=response_schema,
            )
            choice = response.choices[0]
            return CallResult(
                content=choice.message.parsed,
                model=self.model,
                reasoning=_reasoning_content(choice),
            )
        except Exception:
            # Fallback: JSON mode + manual Pydantic parse
            response = self._client.chat.completions.create(
                model=self.model,
                messages=oai_messages,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            reasoning = _reasoning_content(choice)
            raw = choice.message.content or "{}"
            if not reasoning:
                reasoning, raw = _split_reasoning(raw)
            data = json.loads(raw)
            return CallResult(
                content=response_schema(**data),
                model=self.model,
                reasoning=reasoning,
            )
