from openai import OpenAI

from .base import BaseJudge, JudgeResult
from .prompts import CLARITY_SYSTEM_PROMPT, CLARITY_USER_TEMPLATE


class ClarityJudge(BaseJudge):
    def __init__(self, client: OpenAI, model: str) -> None:
        super().__init__(client, model)

    def judge(self, hypothesis: str) -> JudgeResult:
        return self._call(
            system=CLARITY_SYSTEM_PROMPT,
            user=CLARITY_USER_TEMPLATE.format(hypothesis=hypothesis),
        )
