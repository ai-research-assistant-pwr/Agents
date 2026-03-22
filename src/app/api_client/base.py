from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, overload

T = TypeVar("T")


@dataclass
class Message:
    """A single message in a conversation with an LLM."""

    role: str  # "system", "user", or "assistant"
    content: str


class BaseAPIClient(ABC):
    """Abstract base class for LLM API clients.

    Provides a uniform interface for making calls to language models
    regardless of the underlying provider.
    """

    def __init__(self, model: str) -> None:
        self.model = model

    @overload
    def call(self, messages: list[Message], response_schema: None = ...) -> str: ...

    @overload
    def call(self, messages: list[Message], response_schema: type[T]) -> T: ...

    @abstractmethod
    def call(
        self, messages: list[Message], response_schema: type[T] | None = None
    ) -> str | T:
        """Send a list of messages to the LLM and return the response.

        Args:
            messages: Ordered list of conversation messages.
            response_schema: Optional Pydantic model or dataclass. When provided,
                the response is parsed and returned as an instance of this type
                instead of a plain string.

        Returns:
            The model's response as a plain string when response_schema is None,
            or as an instance of response_schema otherwise.
        """
        ...
