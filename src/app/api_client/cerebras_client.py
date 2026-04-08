import json
import os
from typing import TypeVar

from cerebras.cloud.sdk import Cerebras

from app.api_client.base import BaseAPIClient, CallResult, Message

T = TypeVar("T")


def _ensure_no_additional_properties(schema: dict) -> dict:
    """Recursively add ``"additionalProperties": false`` to every object node.

    Cerebras strict mode requires this on every object in the schema, but
    Pydantic's ``model_json_schema()`` does not emit it by default.
    """
    schema = dict(schema)

    if schema.get("type") == "object" or "properties" in schema:
        schema.setdefault("additionalProperties", False)
        if "properties" in schema:
            schema["properties"] = {
                k: _ensure_no_additional_properties(v)
                for k, v in schema["properties"].items()
            }

    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _ensure_no_additional_properties(schema["items"])

    if "$defs" in schema:
        schema["$defs"] = {
            k: _ensure_no_additional_properties(v) for k, v in schema["$defs"].items()
        }

    if "anyOf" in schema:
        schema["anyOf"] = [_ensure_no_additional_properties(s) for s in schema["anyOf"]]

    if "prefixItems" in schema:
        schema["prefixItems"] = [
            _ensure_no_additional_properties(s) for s in schema["prefixItems"]
        ]

    return schema


class CerebrasAPIClient(BaseAPIClient):
    """Cerebras Cloud API client using the Cerebras Cloud SDK.

    Loads the API key from the CEREBRAS_API_KEY environment variable.
    Uses the chat completions endpoint.

    Structured output uses ``response_format={"type": "json_schema", ...}``
    with ``strict: True`` (constrained decoding), which guarantees the
    response matches the provided schema at the token level.

    For Pydantic v2 models the JSON schema is derived via
    ``model_json_schema()``; ``additionalProperties: false`` is added
    recursively as required by Cerebras strict mode.  Non-Pydantic schemas
    fall back to JSON mode with best-effort parsing.
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
                provided, the response is constrained to match the schema via
                Cerebras structured outputs (strict JSON schema mode).

        Returns:
            A CallResult with .model set to the Cerebras model identifier.
            .content is a plain string when response_schema is None, or a
            parsed instance of response_schema otherwise.
        """
        chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        if response_schema is not None:
            if hasattr(response_schema, "model_json_schema"):
                # Pydantic v2: derive JSON schema and enforce via strict mode.
                raw_schema = response_schema.model_json_schema()
                strict_schema = _ensure_no_additional_properties(raw_schema)
                schema_name = getattr(response_schema, "__name__", "response_schema")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=chat_messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": strict_schema,
                        },
                    },
                )
                raw = response.choices[0].message.content
                if not raw:
                    raise RuntimeError(
                        "Cerebras API returned an empty structured response."
                    )
                parsed: T = response_schema.model_validate_json(raw)  # type: ignore[attr-defined]
            else:
                # Fallback for non-Pydantic schemas: JSON mode + manual instantiation.
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
