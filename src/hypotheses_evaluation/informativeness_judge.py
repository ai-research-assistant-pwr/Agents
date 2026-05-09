from app.api_client.base import BaseAPIClient, Message

from .base import BaseJudge, JudgeResponse, JudgeResult
from .prompts import INFORMATIVENESS_SYSTEM_PROMPT, INFORMATIVENESS_USER_TEMPLATE


class InformativenessJudge(BaseJudge):
    """LLM judge that scores the methodological informativeness of a hypothesis.

    A hypothesis scores highly when it describes the method or novel approach
    used to achieve the expected relationship it proposes, rather than merely
    stating that relationship.

    The score reflects the depth of methodological description, on a 0–2 scale:

        0 — hypothesis only proposes a relationship; no methodological detail
        1 — methodology is mentioned but significant aspects are underspecified
        2 — methodology is described in enough detail to leave little doubt

    Args:
        api_client: Any :class:`BaseAPIClient` implementation (e.g.
            ``GoogleAPIClient`` or ``RandomAPIClient``).
    """

    def __init__(self, api_client: BaseAPIClient) -> None:
        super().__init__(api_client)

    def judge(self, hypothesis: str) -> JudgeResult:
        """Evaluate the methodological informativeness of *hypothesis*.

        Args:
            hypothesis: A single hypothesis string to evaluate.

        Returns:
            A :class:`JudgeResult` with ``score`` in ``[0, 2]``, a
            ``reasoning`` string, and the ``model`` that produced the
            judgment.
        """
        messages = [
            Message(role="system", content=INFORMATIVENESS_SYSTEM_PROMPT),
            Message(
                role="user",
                content=INFORMATIVENESS_USER_TEMPLATE.format(hypothesis=hypothesis),
            ),
        ]
        result = self.api_client.call(messages, response_schema=JudgeResponse)
        return JudgeResult(
            score=result.content.score,
            reasoning=result.content.reasoning,
            model=result.model,
        )
