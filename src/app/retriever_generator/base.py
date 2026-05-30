from abc import ABC, abstractmethod

from app.models import ExplorerResult, GeneratorResult


class BaseRetrieverGenerator(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, explorer_output: ExplorerResult
    ) -> GeneratorResult:
        ...
