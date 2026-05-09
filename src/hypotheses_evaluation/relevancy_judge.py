from openai import OpenAI

from .base import BaseJudge, JudgeResult
from .prompts import RELEVANCY_SYSTEM_PROMPT, RELEVANCY_USER_TEMPLATE


class RelevancyJudge(BaseJudge):
    def __init__(self, client: OpenAI, model: str) -> None:
        super().__init__(client, model)

    def judge(self, hypothesis: str, query: str) -> JudgeResult:
        return self._call(
            system=RELEVANCY_SYSTEM_PROMPT,
            user=RELEVANCY_USER_TEMPLATE.format(hypothesis=hypothesis, query=query),
        )
