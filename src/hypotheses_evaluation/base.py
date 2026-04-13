from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from app.api_client.base import BaseAPIClient


class JudgeResponse(BaseModel):
    """Structured output schema shared by all LLM judges."""

    score: int
    reasoning: str


@dataclass
class JudgeResult:
    """Result returned by a single judge evaluation.

    Carries the numeric score, the model's reasoning, and the name of
    the model that produced the judgment — mirroring the CallResult
    pattern used elsewhere in the pipeline.
    """

    score: int
    reasoning: str
    model: str


class BaseJudge(ABC):
    """Abstract base class for LLM-based hypothesis judges.

    All concrete judges wrap a :class:`BaseAPIClient` and expose a
    single ``judge`` method whose signature varies per subclass to
    reflect the inputs each metric requires.
    """

    def __init__(self, api_client: BaseAPIClient) -> None:
        self.api_client = api_client
