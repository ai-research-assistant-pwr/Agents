from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, overload

T = TypeVar("T")


@dataclass
class Message:
    """A single message in a conversation with an LLM."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class CallResult(Generic[T]):
    """Result returned by a single LLM API call.

    Carries both the response content and the name of the model that
    served the call — useful when the client routes dynamically (e.g.
    RandomAPIClient) so callers always know which model was used.
    """

    content: T  # str for plain text, or a parsed schema instance
    model: str  # model identifier that produced this response


class BaseAPIClient(ABC):
    """Abstract base class for LLM API clients.

    Provides a uniform interface for making calls to language models
    regardless of the underlying provider. Implementations are not
    required to hold a fixed model; the chosen model is always reported
    back via CallResult.model.
    """

    @overload
    def call(
        self,
        messages: list[Message],
        response_schema: None = ...,
        temperature: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ) -> CallResult[str]: ...

    @overload
    def call(
        self,
        messages: list[Message],
        response_schema: type[T],
        temperature: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ) -> CallResult[T]: ...

    @abstractmethod
    def call(
        self,
        messages: list[Message],
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ) -> CallResult[str] | CallResult[T]:
        """Send a list of messages to the LLM and return a CallResult.

        Args:
            messages: Ordered list of conversation messages.
            response_schema: Optional Pydantic model or dataclass. When provided,
                the response is parsed and returned as an instance of this type
                instead of a plain string.
            temperature: Sampling temperature (None = provider default).
            presence_penalty: Presence penalty (None = provider default).
            frequency_penalty: Frequency penalty (None = provider default).

        Returns:
            A CallResult whose .content is a plain string when response_schema
            is None, or a parsed instance of response_schema otherwise.
            .model always contains the identifier of the model that was used.
        """
        ...
