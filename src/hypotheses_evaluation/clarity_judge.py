from app.api_client.base import BaseAPIClient, Message

from .base import BaseJudge, JudgeResponse, JudgeResult
from .prompts import CLARITY_SYSTEM_PROMPT, CLARITY_USER_TEMPLATE


class ClarityJudge(BaseJudge):
    """LLM judge that scores the clarity of a hypothesis.

    Clarity is assessed across three independent components:

    * **Conciseness** — no unnecessary words or fragments.
    * **Informativeness** — all information needed for self-contained
      understanding is present.
    * **Ease of understanding** — advanced terms/concepts are well
      explained or are standard in the domain.

    The score reflects how many components are satisfied, on a 0–3 scale:

        0 — hypothesis is syntactically incorrect or not valid English
        1 — exactly one clarity component is fulfilled
        2 — exactly two clarity components are fulfilled
        3 — all three clarity components are fulfilled

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
            A :class:`JudgeResult` with ``score`` in ``[0, 3]``, a
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
