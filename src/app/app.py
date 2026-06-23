import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from app.explorer.base import BaseExplorer, BaseSearchExplorer
from app.generator.base import BaseGenerator
from app.models import GeneratorResult
from app.retriever.base import BaseRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Pipeline:
    """Orchestrator for the scientific hypothesis generation pipeline.

    Runs the full pipeline from an injected configuration mapping:
    search -> explorer -> retriever -> (optional refinement loop) -> generator.
    """

    def __init__(
        self,
        search_explorer: BaseSearchExplorer,
        explorer: BaseExplorer,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        config: Mapping[str, Any],
    ) -> None:
        self.config = dict(config)
        self.search_explorer = search_explorer
        self.explorer = explorer
        self.retriever = retriever
        self.generator = generator

    def run(
        self,
        prompt: str,
        retriever_model_name: str,
        generator_model_name: str,
        retriever_kwargs: dict[str, Any] | None = None,
        generator_kwargs: dict[str, Any] | None = None,
    ) -> GeneratorResult:
        """Run the full hypothesis generation pipeline.

        Args:
            prompt: The user's research prompt / question.

        Returns:
            A GeneratorResult containing the generated hypotheses and metadata.
        """
        pipeline_cfg = self.config.get("pipeline", {})
        retriever_kwargs = retriever_kwargs or {}
        generator_kwargs = generator_kwargs or {}
        save_steps = pipeline_cfg.get("save_steps", False)
        save_dir = self._prepare_save_dir() if save_steps else None

        # Step 1: Search (get paper IDs)
        paper_ids = self.search_explorer.search(prompt)
        if save_dir:
            self._save_step(save_dir, "01_search", {"paper_ids": paper_ids})

        # Step 2: Explorer (expand paper set via graph)
        explorer_output = self.explorer.explore(prompt, paper_ids)
        if save_dir:
            self._save_step(save_dir, "02_explorer", asdict(explorer_output))

        # Step 3: Retriever (initial pass)
        retriever_output = self.retriever.retrieve(
            prompt,
            explorer_output,
            model_name=retriever_model_name,
            **retriever_kwargs,
        )
        if save_dir:
            self._save_step(save_dir, "03_retriever", asdict(retriever_output))

        # Step 4: Refinement loop
        refinement_turns = pipeline_cfg.get("refinement_turns", 0)

        for i in range(refinement_turns):
            feedback = self.generator.provide_feedback(
                prompt,
                retriever_output,
                model_name=generator_model_name,
                **generator_kwargs,
            )
            if save_dir:
                self._save_step(
                    save_dir,
                    f"04_feedback_turn_{i + 1}",
                    {"feedback": feedback},
                )

            retriever_output = self.retriever.refine(
                prompt,
                retriever_output,
                feedback,
                model_name=retriever_model_name,
                **retriever_kwargs,
            )
            if save_dir:
                self._save_step(
                    save_dir,
                    f"04_retriever_refinement_turn_{i + 1}",
                    asdict(retriever_output),
                )

        # Step 5: Generator
        result = self.generator.generate(
            prompt,
            retriever_output,
            model_name=generator_model_name,
            **generator_kwargs,
        )
        if save_dir:
            self._save_step(save_dir, "05_generator", asdict(result))

        # Surface explorer metadata so the API layer can build the knowledge graph.
        result.metadata["explorer"] = explorer_output.metadata
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
