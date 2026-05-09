from openai import OpenAI

from .base import BaseJudge, JudgeResult
from .prompts import GROUNDEDNESS_SYSTEM_PROMPT, GROUNDEDNESS_USER_TEMPLATE


class GroundednessJudge(BaseJudge):
    def __init__(self, client: OpenAI, model: str) -> None:
        super().__init__(client, model)

    def judge(self, hypothesis: str, evidence: str) -> JudgeResult:
        return self._call(
            system=GROUNDEDNESS_SYSTEM_PROMPT,
            user=GROUNDEDNESS_USER_TEMPLATE.format(
                hypothesis=hypothesis, evidence=evidence
            ),
        )
