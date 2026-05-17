from openai import OpenAI

from .base import BaseJudge, JudgeResult
from .prompts import INFORMATIVENESS_SYSTEM_PROMPT, INFORMATIVENESS_USER_TEMPLATE


class InformativenessJudge(BaseJudge):
    def __init__(self, client: OpenAI, model: str) -> None:
        super().__init__(client, model)

    def judge(self, hypothesis: str) -> JudgeResult:
        return self._call(
            system=INFORMATIVENESS_SYSTEM_PROMPT,
            user=INFORMATIVENESS_USER_TEMPLATE.format(hypothesis=hypothesis),
        )
