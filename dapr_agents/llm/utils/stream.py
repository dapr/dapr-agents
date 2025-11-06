from typing import (
    Callable,
    Iterator,
    Optional,
    TypeVar,
)

from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from dapr_agents.types.message import LLMChatCandidateChunk

T = TypeVar("T", bound=BaseModel)


class StreamHandler:
    """
    Handles streaming of chat completion responses, delegating to the
    provider-specific stream processor and optionally validating output
    against Pydantic models.
    """

    @staticmethod
    def process_stream(
        stream: Iterator[ChatCompletionChunk],
        llm_provider: str,
        on_chunk: Optional[Callable],
    ) -> Iterator[LLMChatCandidateChunk]:
        """
        Process a streaming chat completion.

        Args:
            stream:           Iterator of ChatCompletionChunk from OpenAI SDK.
            llm_provider:     Name of the LLM provider (e.g., "openai").
            on_chunk:         Callback fired on every partial LLMChatCandidateChunk.

        Yields:
            LLMChatCandidateChunk: fully-typed chunks, partial and final.
        """

        if llm_provider in ("openai", "nvidia"):
            from dapr_agents.llm.openai.utils import process_openai_stream

            yield from process_openai_stream(
                raw_stream=stream,
                enrich_metadata={"provider": llm_provider},
                on_chunk=on_chunk,
            )
        elif llm_provider == "huggingface":
            from dapr_agents.llm.huggingface.utils import process_hf_stream

            yield from process_hf_stream(
                raw_stream=stream,
                enrich_metadata={"provider": llm_provider},
                on_chunk=on_chunk,
            )
        else:
            raise ValueError(f"Streaming not supported for provider: {llm_provider}")
