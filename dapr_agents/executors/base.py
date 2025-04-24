from dapr_agents.types.executor import ExecutionRequest, CodeSnippet, ExecutionResult
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, ClassVar


class CodeExecutorBase(BaseModel, ABC):
    """Abstract base class for executing code in different environments."""

    SUPPORTED_LANGUAGES: ClassVar[set] = {"python", "sh", "bash"}

    @abstractmethod
    async def execute(self, request: ExecutionRequest) -> List[ExecutionResult]:
        """Executes the provided code snippets and returns results."""
        pass

    def validate_snippets(self, snippets: List[CodeSnippet]) -> bool:
        """Ensures all code snippets are valid before execution."""
        for snippet in snippets:
            if snippet.language not in self.SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {snippet.language}")
        return True
