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
