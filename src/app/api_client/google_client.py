import os
from typing import TypeVar

from google import genai
from google.genai import types

from app.api_client.base import BaseAPIClient, CallResult, Message

T = TypeVar("T")


def _split_parts(response) -> tuple[str, str]:
    """Separate ``thought=True`` parts (reasoning) from answer parts.

    Returns ``(reasoning, answer)`` — both stripped; empty string when absent.
    Gemini thinking models tag internal reasoning parts with ``thought=True``.
    """
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "", ""

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []

    reasoning_parts: list[str] = []
    answer_parts: list[str] = []

    for part in parts:
        text = getattr(part, "text", None) or ""
        if getattr(part, "thought", False):
            reasoning_parts.append(text)
        else:
            answer_parts.append(text)

    reasoning = "\n\n".join(filter(None, reasoning_parts)).strip()
    answer = "\n\n".join(filter(None, answer_parts)).strip()
    return reasoning, answer


class GoogleAPIClient(BaseAPIClient):
    """Google Gemini API client using the google-genai SDK.

    Loads the API key from the GOOGLE_API_KEY environment variable.
    Reasoning content from thinking models (e.g. gemini-2.5-flash with
    thinking enabled) is captured in ``CallResult.reasoning``.
    """

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.model = model
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Set it or add it to a .env file."
            )
        self.client = genai.Client(api_key=api_key)

    def call(  # type: ignore[override]
        self, messages: list[Message], response_schema: type[T] | None = None
    ) -> CallResult[str] | CallResult[T]:
        """Send messages to the Google Gemini model.

        Args:
            messages: Ordered list of conversation messages.
                      Messages with role 'system' are passed as system_instruction.
                      Messages with role 'user' or 'assistant' become conversation turns.
            response_schema: Optional Pydantic model or dataclass. When provided,
                the response is returned as a parsed instance of this type using
                Gemini's native structured output (JSON mode).

        Returns:
            A CallResult with .model set to the Gemini model identifier.
            .content is a plain string when response_schema is None, or a
            parsed instance of response_schema otherwise.
            .reasoning contains the model's thinking trace when available.
        """
        system_parts: list[str] = []
        contents: list[types.Content] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        config = types.GenerateContentConfig()
        if system_parts:
            config.system_instruction = "\n".join(system_parts)

        if response_schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema  # type: ignore[assignment]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,  # type: ignore[arg-type]
            config=config,
        )

        reasoning, answer = _split_parts(response)

        if response_schema is not None:
            if response.parsed is None:
                raise RuntimeError(
                    "Google API returned an empty structured response. "
                    "Check that the model supports structured output."
                )
            return CallResult(  # type: ignore[return-value]
                content=response.parsed,
                model=self.model,
                reasoning=reasoning,
            )

        # Fall back to response.text when no non-thought parts are present
        # (e.g. model without thinking enabled).
        text = answer or response.text
        if text is None:
            raise RuntimeError("Google API returned an empty response.")
        return CallResult(content=text, model=self.model, reasoning=reasoning)
