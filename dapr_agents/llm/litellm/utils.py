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

import logging
from typing import Any, Callable, Dict, Iterator, Optional

from litellm.types.utils import ModelResponse, ModelResponseStream

from dapr_agents.types.message import (
    AssistantMessage,
    FunctionCall,
    FunctionCallChunk,
    LLMChatCandidate,
    LLMChatCandidateChunk,
    LLMChatResponse,
    LLMChatResponseChunk,
    ToolCall,
    ToolCallChunk,
)

logger = logging.getLogger(__name__)


def _translate_tool_calls(message: Any) -> Optional[list[ToolCall]]:
    """Convert LiteLLM message tool calls to normalized tool-call models."""
    if not message.tool_calls:
        return None

    tool_calls = []
    for tool_call in message.tool_calls:
        try:
            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function=FunctionCall(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
            )
        except Exception as error:
            logger.warning(f"Invalid LiteLLM tool_call entry {tool_call}: {error}")
    return tool_calls or None


def _translate_tool_call(tool_call: Any) -> ToolCallChunk:
    """Convert a LiteLLM tool-call object to a normalized chunk model."""
    function = tool_call.function
    return ToolCallChunk(
        index=tool_call.index,
        id=tool_call.id,
        type=tool_call.type,
        function=FunctionCallChunk(
            name=function.name,
            arguments=function.arguments,
        ),
    )


def _get_packet_metadata(
    packet: Any, enrich_metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract LiteLLM packet metadata and merge provider metadata."""
    return {
        "id": packet.id,
        "created": packet.created,
        "model": packet.model,
        "object": getattr(packet, "object", None),
        "service_tier": getattr(packet, "service_tier", None),
        "system_fingerprint": getattr(packet, "system_fingerprint", None),
        "usage": getattr(packet, "usage", None),
        **(enrich_metadata or {}),
    }


def _process_choice_delta(
    choice: Any,
    overall_meta: Dict[str, Any],
    on_chunk: Optional[Callable],
    first_chunk_flag: bool,
) -> Iterator[LLMChatResponseChunk]:
    """Normalize one LiteLLM stream choice."""
    meta = {**overall_meta}
    if first_chunk_flag and "first_chunk" not in meta:
        meta["first_chunk"] = True

    delta = choice.delta
    finish_reason = choice.finish_reason
    if finish_reason in ("stop", "tool_calls"):
        meta["last_chunk"] = True

    function_call = getattr(delta, "function_call", None)
    response_chunk = LLMChatResponseChunk(
        result=LLMChatCandidateChunk(
            content=getattr(delta, "content", None),
            function_call=(
                {
                    "name": function_call.name,
                    "arguments": function_call.arguments,
                }
                if function_call
                else None
            ),
            refusal=getattr(delta, "refusal", None),
            role=getattr(delta, "role", None),
            tool_calls=[
                _translate_tool_call(tool_call)
                for tool_call in (getattr(delta, "tool_calls", None) or [])
            ],
            finish_reason=finish_reason,
            index=choice.index,
            logprobs=getattr(choice, "logprobs", None),
        ),
        metadata=meta,
    )
    if on_chunk:
        on_chunk(response_chunk)
    yield response_chunk


def process_litellm_stream(
    raw_stream: Iterator[ModelResponseStream],
    *,
    enrich_metadata: Optional[Dict[str, Any]] = None,
    on_chunk: Optional[Callable],
) -> Iterator[LLMChatCandidateChunk]:
    """Normalize a native LiteLLM stream into ``LLMChatCandidateChunk`` objects.

    Args:
        raw_stream: Native LiteLLM ``ModelResponseStream`` packets.
        enrich_metadata: Metadata merged into every emitted chunk.
        on_chunk: Optional callback invoked for each choice delta.

    Yields:
        ``LLMChatCandidateChunk``: Normalized response chunks, including empty usage-only chunks.
    """
    enrich_metadata = enrich_metadata or {}
    first_chunk_flag = True

    for packet in raw_stream:
        metadata = _get_packet_metadata(packet, enrich_metadata)
        if choices := packet.choices:
            yield from _process_choice_delta(
                choices[0], metadata, on_chunk, first_chunk_flag
            )
            first_chunk_flag = False
        else:
            logger.debug(f"Yielding final LiteLLM usage-only packet: {packet}")
            yield LLMChatResponseChunk(
                result=LLMChatCandidateChunk(),
                metadata=metadata,
            )


def process_litellm_chat_response(response: ModelResponse) -> LLMChatResponse:
    """Normalize a native LiteLLM ``ModelResponse`` into a unified ``LLMChatResponse``.

    Args:
        response: Native LiteLLM response returned for a non-streaming call.

    Returns:
        ``LLMChatResponse``: A normalized response containing candidates and provider metadata.
    """
    candidates = []
    for choice in response.choices:
        message = choice.message
        tool_calls = _translate_tool_calls(message)
        function_call = getattr(message, "function_call", None)
        candidates.append(
            LLMChatCandidate(
                message=AssistantMessage(
                    content=message.content,
                    refusal=getattr(message, "refusal", None),
                    tool_calls=tool_calls,
                    function_call=(
                        FunctionCall(
                            name=function_call.name or "",
                            arguments=function_call.arguments or "",
                        )
                        if function_call
                        else None
                    ),
                ),
                finish_reason=choice.finish_reason,
                index=choice.index,
                logprobs=getattr(choice, "logprobs", None),
            )
        )

    return LLMChatResponse(
        results=candidates,
        metadata={
            "provider": "litellm",
            "id": response.id,
            "model": response.model,
            "object": getattr(response, "object", None),
            "usage": getattr(response, "usage", None),
            "created": response.created,
            "service_tier": getattr(response, "service_tier", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
        },
    )
