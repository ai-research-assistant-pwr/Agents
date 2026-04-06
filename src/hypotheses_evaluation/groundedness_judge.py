from app.api_client.base import BaseAPIClient, Message

from .base import BaseJudge, JudgeResponse, JudgeResult
from .prompts import GROUNDEDNESS_SYSTEM_PROMPT, GROUNDEDNESS_USER_TEMPLATE


class GroundednessJudge(BaseJudge):
    """LLM judge that scores how well a hypothesis is grounded in evidence.

    The judge checks whether the concepts stated in the hypothesis can be
    traced back to the evidence summary produced by the retriever.  It
    returns a score on a 0–4 integer scale:

        0 — hypothesis is not related at all to the provided evidence
        1 — minority of concepts are present in the evidence
        2 — about half of the concepts are present in the evidence
        3 — majority of concepts are present in the evidence
        4 — all concepts are present in the evidence

    Args:
        api_client: Any :class:`BaseAPIClient` implementation (e.g.
            ``GoogleAPIClient`` or ``RandomAPIClient``).
    """

    def __init__(self, api_client: BaseAPIClient) -> None:
        super().__init__(api_client)

    def judge(self, hypothesis: str, evidence: str) -> JudgeResult:
        """Evaluate the groundedness of *hypothesis* against *evidence*.

        Args:
            hypothesis: A single hypothesis string to evaluate.
            evidence: The retriever's evidence summary that was available
                to the hypothesis generator (``RetrieverResult.content``).

        Returns:
            A :class:`JudgeResult` with ``score`` in ``[0, 4]``, a
            ``reasoning`` string, and the ``model`` that produced the
            judgment.
        """
        messages = [
            Message(role="system", content=GROUNDEDNESS_SYSTEM_PROMPT),
            Message(
                role="user",
                content=GROUNDEDNESS_USER_TEMPLATE.format(
                    hypothesis=hypothesis,
                    evidence=evidence,
                ),
            ),
        ]
        result = self.api_client.call(messages, response_schema=JudgeResponse)
        return JudgeResult(
            score=result.content.score,
            reasoning=result.content.reasoning,
            model=result.model,
        )
