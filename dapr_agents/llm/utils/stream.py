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

from typing import (
    Any,
    Callable,
    Iterator,
    Optional,
    TypeVar,
    Union,
)

from pydantic import BaseModel

from dapr_agents.types.message import LLMChatCandidateChunk, LLMChatResponseChunk

T = TypeVar("T", bound=BaseModel)


class StreamHandler:
    """
    Handles streaming of chat completion responses, delegating to the
    provider-specific stream processor and optionally validating output
    against Pydantic models.
    """

    @staticmethod
    def process_stream(
        stream: Any,
        llm_provider: str,
        on_chunk: Optional[Callable] = None,
    ) -> Iterator[Union[LLMChatCandidateChunk, LLMChatResponseChunk]]:
        """
        Process a streaming chat completion.

        Args:
            stream:           Raw stream object or iterator from the provider SDK.
            llm_provider:     Name of the LLM provider (e.g., "openai", "anthropic").
            on_chunk:         Callback fired on every partial chunk.

        Yields:
            LLMChatCandidateChunk | LLMChatResponseChunk: fully-typed chunks, partial and final.
        """

        if llm_provider in ("openai", "nvidia", "litellm", "iflytek"):
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
        elif llm_provider in ("anthropic", "claude"):
            from dapr_agents.llm.anthropic.utils import process_anthropic_stream

            yield from process_anthropic_stream(
                raw_stream=stream,
                enrich_metadata={"provider": llm_provider},
                on_chunk=on_chunk,
            )
        else:
            raise ValueError(f"Streaming not supported for provider: {llm_provider}")
