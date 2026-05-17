from concurrent.futures import ThreadPoolExecutor
from statistics import median

from openai import OpenAI

from .base import BaseJudge, JudgeResult
from .prompts import RELEVANCY_SYSTEM_PROMPT, RELEVANCY_USER_TEMPLATE

NUM_JUDGES = 1


class RelevancyJudge(BaseJudge):
    def __init__(
        self, 
        client: OpenAI, 
        model: str = "gpt-5.4-mini", 
        reasoning: dict = {"effort": "low"}, 
        num_judges: int = NUM_JUDGES
    ) -> None:
        super().__init__(client, model, reasoning)
        self.num_judges = num_judges

    def judge(self, hypothesis: str, query: str) -> JudgeResult:
        user = RELEVANCY_USER_TEMPLATE.format(hypothesis=hypothesis, query=query)

        def _single_call(_: int) -> JudgeResult:
            return self._call(system=RELEVANCY_SYSTEM_PROMPT, user=user)

        with ThreadPoolExecutor(max_workers=self.num_judges) as executor:
            results = list(executor.map(_single_call, range(self.num_judges)))

        scores = [r.score for r in results]
        final_score = int(median(scores))

        return JudgeResult(
            score=final_score,
            model=self.model,
        )
