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

import dataclasses
import logging
from typing import Any, Callable, Dict, Iterator, Optional

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from dapr_agents.llm.utils.stream import (
    extract_packet_metadata as _get_packet_metadata,
    process_choice_delta as _process_choice_delta,
    process_choice_delta_stream,
)
from dapr_agents.types.message import (
    AssistantMessage,
    FunctionCall,
    LLMChatCandidate,
    LLMChatResponse,
    LLMChatResponseChunk,
    ToolCall,
)

logger = logging.getLogger(__name__)


def process_openai_stream(
    raw_stream: Iterator[ChatCompletionChunk],
    *,
    enrich_metadata: Optional[Dict[str, Any]] = None,
    on_chunk: Optional[Callable] = None,
) -> Iterator[LLMChatResponseChunk]:
    """Normalize OpenAI streaming chat into LLMChatResponseChunk objects,
    accumulating buffers per choice and yielding both partial and final chunks.

    Args:
        raw_stream: Iterator from client.chat.completions.create(..., stream=True)
        enrich_metadata: Extra key/value pairs to merge into each chunk.metadata
        on_chunk:   Callback fired on every partial delta (token, function, tool)

    Yields:
        LLMChatResponseChunk for every partial and final piece, in stream order
    """
    yield from process_choice_delta_stream(
        raw_stream=raw_stream,
        enrich_metadata=enrich_metadata,
        on_chunk=on_chunk,
    )


def process_openai_chat_response(openai_response: ChatCompletion) -> LLMChatResponse:
    """
    Convert an OpenAI ChatCompletion into our unified LLMChatResponse.

    This function:
      - Safely extracts each choice (skipping malformed ones)
      - Builds an AssistantMessage with content/refusal/tool_calls/function_call
      - Wraps into LLMChatCandidate (including index & logprobs)
      - Collects provider metadata

    Args:
        openai_response: A Pydantic ChatCompletion from the OpenAI SDK.

    Returns:
        LLMChatResponse: Contains a list of candidates and a metadata dict.
    """
    # 1) Turn into plain dict
    try:
        if hasattr(openai_response, "model_dump"):
            resp = openai_response.model_dump()
        elif hasattr(openai_response, "to_dict"):
            resp = openai_response.to_dict()
        elif dataclasses.is_dataclass(openai_response):
            resp = dataclasses.asdict(openai_response)
        else:
            resp = dict(openai_response)
    except Exception:
        logger.exception("Failed to serialize OpenAI chat response")
        resp = {}

    candidates = []
    for choice in resp.get("choices", []):
        if "message" not in choice:
            logger.warning(f"Skipping choice missing 'message': {choice}")
            continue

        msg = choice["message"]
        # 2) Build tool_calls list if present
        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = []
            for tc in msg["tool_calls"]:
                try:
                    tool_calls.append(
                        ToolCall(
                            id=tc["id"],
                            type=tc["type"],
                            function=FunctionCall(
                                name=tc["function"]["name"],
                                arguments=tc["function"]["arguments"],
                            ),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Invalid tool_call entry {tc}: {e}")

        # 3) Build function_call if present
        function_call = None
        if fc := msg.get("function_call"):
            function_call = FunctionCall(
                name=fc.get("name", ""),
                arguments=fc.get("arguments", ""),
            )

        # 4) Assemble AssistantMessage
        assistant_message = AssistantMessage(
            content=msg.get("content"),
            refusal=msg.get("refusal"),
            tool_calls=tool_calls,
            function_call=function_call,
        )

        # 5) Build candidate, including index & logprobs
        candidate = LLMChatCandidate(
            message=assistant_message,
            finish_reason=choice.get("finish_reason"),
            index=choice.get("index"),
            logprobs=choice.get("logprobs"),
        )
        candidates.append(candidate)

    # 6) Metadata: include provider tag
    metadata: Dict[str, Any] = {
        "provider": "openai",
        "id": resp.get("id"),
        "model": resp.get("model"),
        "object": resp.get("object"),
        "usage": resp.get("usage"),
        "created": resp.get("created"),
    }

    return LLMChatResponse(results=candidates, metadata=metadata)


__all__ = [
    "process_openai_stream",
    "process_openai_chat_response",
    "_get_packet_metadata",
    "_process_choice_delta",
]
