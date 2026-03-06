#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
