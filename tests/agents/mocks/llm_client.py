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

from dapr_agents.llm import OpenAIChatClient
from dapr_agents.types.message import (
    LLMChatResponse,
    LLMChatCandidate,
    AssistantMessage,
)


class MockLLMClient(OpenAIChatClient):
    """Mock LLM client for testing that passes type validation."""

    def __init__(self, **kwargs):
        super().__init__(model=kwargs.get("model", "gpt-4o"), api_key="mock-api-key")
        self.prompt_template = kwargs.get("prompt_template", None)

    def generate(
        self,
        messages=None,
        *,
        input_data=None,
        model=None,
        tools=None,
        response_format=None,
        structured_mode="json",
        stream=False,
        **kwargs,
    ):
        return LLMChatResponse(
            results=[
                LLMChatCandidate(
                    message=AssistantMessage(
                        content="This is a mock response from the LLM client."
                    ),
                    finish_reason="stop",
                )
            ]
        )
