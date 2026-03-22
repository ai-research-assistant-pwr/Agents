from app.config import load_config
from app.explorer.base import BaseExplorer
from app.retriever.base import BaseRetriever
from app.generator.base import BaseGenerator


class App:
    """Orchestrator for the scientific hypothesis generation pipeline.

    Reads configuration from a YAML file and runs the full pipeline:
    explorer -> retriever -> generator.
    """

    def __init__(
        self,
        explorer: BaseExplorer,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        config_path: str = "config/app/config.yaml",
    ) -> None:
        self.config = load_config(config_path)
        self.explorer = explorer
        self.retriever = retriever
        self.generator = generator

    def run(self, prompt: str) -> list[str]:
        """Run the full hypothesis generation pipeline.

        Args:
            prompt: The user's research prompt / question.

        Returns:
            A list of generated hypothesis strings.
        """
        explorer_output = self.explorer.explore(prompt)
        retriever_output = self.retriever.retrieve(prompt, explorer_output)
        hypotheses = self.generator.generate(prompt, retriever_output)
        return hypotheses
