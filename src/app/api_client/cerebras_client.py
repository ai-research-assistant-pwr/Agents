import json
import os
from typing import TypeVar

from cerebras.cloud.sdk import Cerebras

from app.api_client.base import BaseAPIClient, CallResult, Message

T = TypeVar("T")


class CerebrasAPIClient(BaseAPIClient):
    """Cerebras Cloud API client using the Cerebras Cloud SDK.

    Loads the API key from the CEREBRAS_API_KEY environment variable.
    Uses the chat completions endpoint, which is compatible with the
    OpenAI Chat Completions interface.

    Structured output is achieved via JSON mode (``response_format={"type":
    "json_object"}``). The raw JSON is then validated against the provided
    response_schema:

    * If response_schema exposes ``model_validate_json`` (Pydantic v2), that
      method is used directly.
    * Otherwise the JSON is decoded and the schema is instantiated with
      keyword arguments (e.g. plain dataclasses).
    """

    def __init__(self, model: str = "llama-3.3-70b") -> None:
        self.model = model
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if api_key is None:
            raise ValueError(
                "CEREBRAS_API_KEY environment variable is not set. "
                "Set it or add it to a .env file."
            )
        self.client = Cerebras(api_key=api_key)

    def call(  # type: ignore[override]
        self, messages: list[Message], response_schema: type[T] | None = None
    ) -> CallResult[str] | CallResult[T]:
        """Send messages to a Cerebras model.

        Args:
            messages: Ordered list of conversation messages. All roles
                      ('system', 'user', 'assistant') are forwarded as-is to
                      the chat completions endpoint.
            response_schema: Optional Pydantic model or dataclass. When
                provided, JSON mode is enabled and the response is parsed into
                an instance of this type.

        Returns:
            A CallResult with .model set to the Cerebras model identifier.
            .content is a plain string when response_schema is None, or a
            parsed instance of response_schema otherwise.
        """
        chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        if response_schema is not None:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            if not raw:
                raise RuntimeError(
                    "Cerebras API returned an empty structured response."
                )
            if hasattr(response_schema, "model_validate_json"):
                # Pydantic v2
                parsed: T = response_schema.model_validate_json(raw)  # type: ignore[attr-defined]
            else:
                parsed = response_schema(**json.loads(raw))
            return CallResult(content=parsed, model=self.model)  # type: ignore[return-value]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Cerebras API returned an empty response.")
        return CallResult(content=content, model=self.model)
