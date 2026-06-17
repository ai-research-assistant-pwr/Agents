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

    def __init__(self, text: str | None = None) -> None:
        self.text = text if text is not None else self.DEFAULT_TEXT

    def explore(self, prompt: str, paper_ids: list[str] | None = None) -> ExplorerResult:
        """Return constant text regardless of the prompt.

        Args:
            prompt: The user's research prompt (ignored).

        Returns:
            An ExplorerResult with the constant text and metadata noting this
            is a placeholder.
        """
        return ExplorerResult(
            content=self.text,
            metadata={"source": "const_explorer", "prompt_received": prompt},
        )
