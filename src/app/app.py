import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.config import PROJECT_ROOT, load_config
from app.explorer.base import BaseExplorer
from app.generator.base import BaseGenerator
from app.models import ExplorerResult, GeneratorResult
from app.retriever.base import BaseRetriever


class App:
    """Orchestrator for the scientific hypothesis generation pipeline.

    Reads configuration from a YAML file and runs the full pipeline:
    explorer -> retriever -> (optional refinement loop) -> generator.
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

    def run(self, prompt: str, context: str | None = None) -> GeneratorResult:
        """Run the full hypothesis generation pipeline.

        Args:
            prompt: The user's research prompt / question.
            context: Optional pre-formed context string to pass directly to the
                retriever, bypassing the explorer entirely. When provided the
                explorer is not called and this string is wrapped in an
                ExplorerResult instead.

        Returns:
            A GeneratorResult containing the generated hypotheses and metadata.
        """
        pipeline_cfg = self.config.get("pipeline", {})
        save_steps = pipeline_cfg.get("save_steps", False)
        save_dir = self._prepare_save_dir() if save_steps else None

        # Step 1: Explorer (skipped when context is provided directly)
        if context is not None:
            explorer_output = ExplorerResult(
                content=context,
                metadata={"source": "provided_context"},
            )
        else:
            explorer_output = self.explorer.explore(prompt)
        if save_dir:
            self._save_step(save_dir, "01_explorer", asdict(explorer_output))

        # Step 2: Retriever (initial pass)
        retriever_output = self.retriever.retrieve(prompt, explorer_output)
        if save_dir:
            self._save_step(save_dir, "02_retriever", asdict(retriever_output))

        # Step 3: Refinement loop
        refinement_turns = pipeline_cfg.get("refinement_turns", 0)

        for i in range(refinement_turns):
            feedback = self.generator.provide_feedback(prompt, retriever_output)
            if save_dir:
                self._save_step(
                    save_dir,
                    f"03_feedback_turn_{i + 1}",
                    {"feedback": feedback},
                )

            retriever_output = self.retriever.refine(prompt, retriever_output, feedback)
            if save_dir:
                self._save_step(
                    save_dir,
                    f"03_retriever_refinement_turn_{i + 1}",
                    asdict(retriever_output),
                )

        # Step 4: Generator
        result = self.generator.generate(prompt, retriever_output)
        if save_dir:
            self._save_step(save_dir, "04_generator", asdict(result))

        return result

    def _prepare_save_dir(self) -> Path:
        """Create a timestamped directory for saving pipeline step outputs."""
        save_root = self.config.get("pipeline", {}).get("save_dir", "outputs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = PROJECT_ROOT / save_root / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    @staticmethod
    def _save_step(save_dir: Path, name: str, data: dict) -> None:
        """Save a pipeline step result as a JSON file."""
        path = save_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
