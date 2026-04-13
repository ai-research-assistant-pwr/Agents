from app.api_client.base import BaseAPIClient, Message

from .base import BaseJudge, JudgeResponse, JudgeResult
from .prompts import RELEVANCY_SYSTEM_PROMPT, RELEVANCY_USER_TEMPLATE


class RelevancyJudge(BaseJudge):
    """LLM judge that scores how relevant a hypothesis is to the user's query.

    The judge checks how many of the concepts and inquiries expressed in
    the original user query are addressed by the hypothesis.  It returns
    a score on a 0–4 integer scale:

        0 — hypothesis is not related at all to the user query
        1 — hypothesis addresses some of the query's concepts/inquiries
        2 — hypothesis addresses about half of the query's concepts/inquiries
        3 — hypothesis addresses most of the query's concepts/inquiries
        4 — hypothesis addresses all of the query's concepts/inquiries

    Args:
        api_client: Any :class:`BaseAPIClient` implementation (e.g.
            ``GoogleAPIClient`` or ``RandomAPIClient``).
    """

    def __init__(self, api_client: BaseAPIClient) -> None:
        super().__init__(api_client)

    def judge(self, hypothesis: str, query: str) -> JudgeResult:
        """Evaluate the relevancy of *hypothesis* with respect to *query*.

        Args:
            hypothesis: A single hypothesis string to evaluate.
            query: The original user query that initiated the pipeline run.

        Returns:
            A :class:`JudgeResult` with ``score`` in ``[0, 4]``, a
            ``reasoning`` string, and the ``model`` that produced the
            judgment.
        """
        messages = [
            Message(role="system", content=RELEVANCY_SYSTEM_PROMPT),
            Message(
                role="user",
                content=RELEVANCY_USER_TEMPLATE.format(
                    hypothesis=hypothesis,
                    query=query,
                ),
            ),
        ]
        result = self.api_client.call(messages, response_schema=JudgeResponse)
        return JudgeResult(
            score=result.content.score,
            reasoning=result.content.reasoning,
            model=result.model,
        )
