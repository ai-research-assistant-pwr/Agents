from abc import ABC
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel


class JudgeResponse(BaseModel):
    """Structured output schema shared by all LLM judges."""

    score: int


@dataclass
class JudgeResult:
    score: int
    model: str


class BaseJudge(ABC):
    def __init__(self, client: OpenAI, model: str = "gpt-5.4-mini", reasoning: dict = {"effort": "low"}) -> None:
        self.client = client
        self.model = model
        self.reasoning = reasoning

    def _call(self, system: str, user: str) -> JudgeResult:
        response = self.client.responses.parse(
            model=self.model,
            instructions=system,
            input=[{"role": "user", "content": user}],
            text_format=JudgeResponse,
            reasoning=self.reasoning,
        )
        parsed: JudgeResponse = response.output_parsed
        return JudgeResult(score=parsed.score, model=self.model)
