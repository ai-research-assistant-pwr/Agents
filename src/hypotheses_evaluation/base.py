from abc import ABC
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel


class JudgeResponse(BaseModel):
    """Structured output schema shared by all LLM judges."""

    score: int
    reasoning: str


@dataclass
class JudgeResult:
    score: int
    reasoning: str
    model: str


class BaseJudge(ABC):
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def _call(self, system: str, user: str) -> JudgeResult:
        response = self.client.responses.parse(
            model=self.model,
            instructions=system,
            input=[{"role": "user", "content": user}],
            text_format=JudgeResponse,
            reasoning={"effort": "medium"},
        )
        parsed: JudgeResponse = response.output_parsed
        return JudgeResult(
            score=parsed.score, reasoning=parsed.reasoning, model=self.model
        )
