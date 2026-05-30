from app.api_client.base import BaseAPIClient, Message

from .base import BaseJudge, JudgeResponse, JudgeResult
from .prompts import DIVERSITY_SYSTEM_PROMPT, DIVERSITY_USER_TEMPLATE


class DiversityJudge(BaseJudge):
    """LLM judge that scores the conceptual diversity of a set of hypotheses.

    The judge evaluates the set as a whole and determines the degree of
    conceptual, methodological, and paradigmatic variance among the
    hypotheses.  It returns a score on a 0–4 integer scale:

        0 — No Diversity (Cloning): hypotheses are virtually identical.
        1 — Superficial Diversity: same core method, minor detail changes.
        2 — Parametric Diversity: same methodology, different key variables.
        3 — Methodological Diversity: different mechanisms/algorithms.
        4 — Paradigmatic Diversity: complete conceptual spread across
            different domains or opposing paradigms.

    Args:
        api_client: Any :class:`BaseAPIClient` implementation (e.g.
            ``GoogleAPIClient`` or ``RandomAPIClient``).
    """

    def __init__(self, api_client: BaseAPIClient) -> None:
        super().__init__(api_client)

    def judge(self, hypotheses: list[str], query: str) -> JudgeResult:
        """Evaluate the conceptual diversity of the *hypotheses* set.

        Args:
            hypotheses: List of hypothesis strings to evaluate as a set.
            query: The original user query that initiated the pipeline run.

        Returns:
            A :class:`JudgeResult` with ``score`` in ``[0, 4]``, a
            ``reasoning`` string, and the ``model`` that produced the
            judgment.
        """
        numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hypotheses))
        messages = [
            Message(role="system", content=DIVERSITY_SYSTEM_PROMPT),
            Message(
                role="user",
                content=DIVERSITY_USER_TEMPLATE.format(
                    query=query,
                    hypotheses_list=numbered,
                ),
            ),
        ]
        result = self.api_client.call(messages, response_schema=JudgeResponse)
        return JudgeResult(
            score=result.content.score,
            reasoning=result.content.reasoning,
            model=result.model,
        )
