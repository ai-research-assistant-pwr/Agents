from app.api_client.base import BaseAPIClient, Message

from .base import BaseJudge, JudgeResponse, JudgeResult
from .prompts import CLARITY_SYSTEM_PROMPT, CLARITY_USER_TEMPLATE


class ClarityJudge(BaseJudge):
    """LLM judge that scores the clarity of a hypothesis.

    Clarity is assessed by checking whether all non-obvious concepts that
    are necessary to understand the hypothesis are explained within it.
    The intended audience is a domain expert, so standard field-specific
    concepts do not require explanation.

    The score reflects whether this condition is met, on a 0–1 scale:

        0 — one or more necessary non-obvious concepts are left unexplained
        1 — all concepts necessary for understanding are either explained or
            can be assumed as common knowledge for a domain expert

    Args:
        api_client: Any :class:`BaseAPIClient` implementation (e.g.
            ``GoogleAPIClient`` or ``RandomAPIClient``).
    """

    def __init__(self, api_client: BaseAPIClient) -> None:
        super().__init__(api_client)

    def judge(self, hypothesis: str) -> JudgeResult:
        """Evaluate the clarity of *hypothesis*.

        Args:
            hypothesis: A single hypothesis string to evaluate.

        Returns:
            A :class:`JudgeResult` with ``score`` in ``[0, 1]``, a
            ``reasoning`` string, and the ``model`` that produced the
            judgment.
        """
        messages = [
            Message(role="system", content=CLARITY_SYSTEM_PROMPT),
            Message(
                role="user",
                content=CLARITY_USER_TEMPLATE.format(hypothesis=hypothesis),
            ),
        ]
        result = self.api_client.call(messages, response_schema=JudgeResponse)
        return JudgeResult(
            score=result.content.score,
            reasoning=result.content.reasoning,
            model=result.model,
        )
