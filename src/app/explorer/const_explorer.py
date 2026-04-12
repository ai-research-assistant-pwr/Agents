from app.explorer.base import BaseExplorer
from app.models import ExplorerResult


class ConstExplorer(BaseExplorer):
    """A placeholder explorer that returns a constant text.

    Useful for testing the pipeline when no knowledge graph is available.
    """

    DEFAULT_TEXT = (
        "No knowledge graph connected. This is placeholder context.\n"
        "In a real scenario, this component would query a knowledge graph "
        "and return relevant papers, entities, and relationships."
    )

    CHUNK_ID = "const_chunk_0"

    def __init__(self, text: str | None = None) -> None:
        self.text = text if text is not None else self.DEFAULT_TEXT

    def explore(self, prompt: str) -> ExplorerResult:
        """Return a single constant chunk regardless of the prompt.

        Args:
            prompt: The user's research prompt (ignored).

        Returns:
            An ExplorerResult with one placeholder chunk ID and metadata
            noting this is a placeholder.
        """
        chunk_id = self.CHUNK_ID
        return ExplorerResult(
            chunk_ids=[chunk_id],
            chunks={chunk_id: {"content": self.text}},
            metadata={"source": "const_explorer", "prompt_received": prompt},
        )
