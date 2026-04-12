from app.api_client.base import BaseAPIClient, Message
from app.models import ExplorerResult, RetrieverResult
from app.retriever.base import BaseRetriever
from app.retriever.prompts import (
    REFINE_SYSTEM_PROMPT,
    REFINE_USER_TEMPLATE,
    RETRIEVER_SYSTEM_PROMPT,
    RETRIEVE_USER_TEMPLATE,
)


def _parse_chunks(explorer_output: ExplorerResult) -> str:
    """Convert raw chunk data from an ExplorerResult into formatted text.

    Iterates over ``explorer_output.chunk_ids`` in order and builds a
    human-readable representation for each chunk.  Fields recognised by
    name (``title``, ``type``, ``paperId``, ``distance``) are used to
    construct a header line; the ``content`` field provides the body.
    Unknown extra fields are ignored.

    Args:
        explorer_output: The result returned by an explorer.

    Returns:
        A single string with sections separated by ``"\\n\\n---\\n\\n"``, or
        ``"No relevant materials found."`` when no chunks are present.
    """
    sections: list[str] = []
    for chunk_id in explorer_output.chunk_ids:
        chunk = explorer_output.chunks.get(chunk_id, {})

        # Build an optional header from well-known metadata fields.
        type_ = chunk.get("type", "")
        title = chunk.get("title", "")
        paper_id = chunk.get("paperId", "")
        distance = chunk.get("distance")

        header_parts = []
        if type_ or title or paper_id:
            dist_str = f" (distance={distance:.4f})" if distance is not None else ""
            header_parts.append(
                f"[{type_}] {title or 'Unknown'} (id={paper_id}){dist_str}"
            )

        body = chunk.get("content", "")
        section = "\n".join(header_parts + [body]) if header_parts else body
        sections.append(section)

    return "\n\n---\n\n".join(sections) if sections else "No relevant materials found."


class APILLMRetriever(BaseRetriever):
    """Retriever that uses an LLM API to filter and rank explorer output."""

    def __init__(self, api_client: BaseAPIClient) -> None:
        self.api_client = api_client

    def retrieve(self, prompt: str, explorer_output: ExplorerResult) -> RetrieverResult:
        """Parse explorer chunks into text, then call the LLM to extract the
        most relevant information.

        Args:
            prompt: The user's research prompt / question.
            explorer_output: Result returned by the explorer (chunk IDs +
                raw chunk data).

        Returns:
            A RetrieverResult with the LLM-filtered content.
        """
        parsed_text = _parse_chunks(explorer_output)

        user_content = RETRIEVE_USER_TEMPLATE.format(
            prompt=prompt,
            explorer_output=parsed_text,
        )
        messages = [
            Message(role="system", content=RETRIEVER_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        result = self.api_client.call(messages)

        return RetrieverResult(
            content=result.content,
            metadata={
                "source": "api_llm_retriever",
                "model": result.model,
            },
        )

    def refine(
        self,
        prompt: str,
        current_context: RetrieverResult,
        generator_feedback: str,
    ) -> RetrieverResult:
        """Call the LLM to refine the retrieved context based on generator feedback.

        Args:
            prompt: The user's research prompt / question.
            current_context: The current retriever result being refined.
            generator_feedback: Feedback string from the generator.

        Returns:
            A refined RetrieverResult.
        """
        user_content = REFINE_USER_TEMPLATE.format(
            prompt=prompt,
            current_context=current_context.content,
            generator_feedback=generator_feedback,
        )
        messages = [
            Message(role="system", content=REFINE_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        result = self.api_client.call(messages)

        return RetrieverResult(
            content=result.content,
            metadata={
                "source": "api_llm_retriever",
                "model": result.model,
                "step": "refine",
            },
        )
