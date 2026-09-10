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

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from dapr_agents.types.message import (
    AssistantMessage,
    FunctionCall,
    LLMChatCandidate,
    LLMChatCandidateChunk,
    LLMChatResponseChunk,
    LLMChatResponse,
    ToolCall,
    ToolCallChunk,
)

logger = logging.getLogger(__name__)


# Helper function to handle metadata extraction
def _get_packet_metadata(
    packet: ChatCompletionChunk, enrich_metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract metadata from OpenAI packet and merge with enrich_metadata.

    Args:
        packet (ChatCompletionChunk): The OpenAI packet from which to extract metadata.
        enrich_metadata (Optional[Dict[str, Any]]): Additional metadata to merge with the extracted metadata.

    Returns:
        Dict[str, Any]: The merged metadata dictionary.
    """

    try:
        return {
            "id": packet.id,
            "created": packet.created,
            "model": packet.model,
            "object": packet.object,
            "service_tier": packet.service_tier,
            "system_fingerprint": packet.system_fingerprint,
            "usage": packet.usage,
            **(enrich_metadata or {}),
        }
    except Exception as e:
        logger.error(f"Failed to parse packet: {e}", exc_info=True)
        return {}


# Helper function to process each choice delta (content, function call, tool call, finish reason)
def _process_choice_delta(
    choice: Any,
    overall_meta: Dict[str, Any],
    on_chunk: Optional[Callable],
    first_chunk_flag: bool,
) -> Iterator[LLMChatResponseChunk]:
    """
    Process each choice delta and yield corresponding chunks.

    Args:
        choice (Dict[str, Any]): The choice delta from OpenAI response.
        overall_meta (Dict[str, Any]): Overall metadata to include in chunks.
        on_chunk (Optional[Callable]): Callback for each chunk.
        first_chunk_flag (bool): Flag indicating if this is the first chunk.

    Yields:
        LLMChatResponseChunk: The processed chunk with content, function call, tool calls,
    """
    # Make an immutable snapshot for this single chunk
    meta = {**overall_meta}

    # mark first_chunk exactly once
    if first_chunk_flag and "first_chunk" not in meta:
        meta["first_chunk"] = True

    # Extract initial properties from choice
    delta = choice.delta
    idx = choice.index
    finish_reason = choice.finish_reason
    logprobs = choice.logprobs

    # Set additional metadata
    if finish_reason in ("stop", "tool_calls"):
        meta["last_chunk"] = True

    # Process content delta
    content = delta.content
    # OpenAI's legacy ``function_call`` field is retained for compatibility;
    # current responses represent function invocations through ``tool_calls``.
    function_call = (
        {
            "name": delta.function_call.name,
            "arguments": delta.function_call.arguments,
        }
        if delta.function_call
        else None
    )
    refusal = delta.refusal
    role = delta.role

    # Process tool calls
    chunk_tool_calls = [
        ToolCallChunk(
            index=tc.index,
            id=tc.id,
            type=tc.type,
            function={
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        )
        for tc in (delta.tool_calls or [])
    ]

    # Initialize LLMChatResponseChunk
    response_chunk = LLMChatResponseChunk(
        result=LLMChatCandidateChunk(
            content=content,
            function_call=function_call,
            refusal=refusal,
            role=role,
            tool_calls=chunk_tool_calls,
            finish_reason=finish_reason,
            index=idx,
            logprobs=logprobs,
        ),
        metadata=meta,
    )
    # Process chunk with on_chunk callback
    if on_chunk:
        on_chunk(response_chunk)
    # Yield LLMChatResponseChunk
    yield response_chunk


# Main function to process OpenAI streaming response
def process_openai_stream(
    raw_stream: Iterator[ChatCompletionChunk],
    *,
    enrich_metadata: Optional[Dict[str, Any]] = None,
    on_chunk: Optional[Callable],
) -> Iterator[LLMChatCandidateChunk]:
    """
    Normalize OpenAI streaming chat into LLMChatCandidateChunk objects,
    accumulating buffers per choice and yielding both partial and final chunks.

    Args:
        raw_stream: Iterator from client.chat.completions.create(..., stream=True)
        enrich_metadata: Extra key/value pairs to merge into each chunk.metadata
        on_chunk:   Callback fired on every partial delta (token, function, tool)

    Yields:
        LLMChatCandidateChunk for every partial and final piece, in stream order
    """
    enrich_metadata = enrich_metadata or {}
    overall_meta: Dict[str, Any] = {}

    # Track if we are in the first chunk
    first_chunk_flag = True

    for packet in raw_stream:
        # Capture overall metadata from the packet
        overall_meta = _get_packet_metadata(packet, enrich_metadata)

        # Process each choice in this packet
        if choices := packet.choices:
            # Process the first choice in the packet
            choice = choices[0]
            yield from _process_choice_delta(
                choice, overall_meta, on_chunk, first_chunk_flag
            )
            # Set first_chunk_flag to False after processing the first choice
            first_chunk_flag = False
        else:
            logger.debug(
                f"Yielding final packet without 'choices' (usage-only): {packet}"
            )
            # Final usage-only packet (empty ``choices``) sent by OpenAI when
            # ``stream_options.include_usage`` is on. ``result`` is required on
            # LLMChatResponseChunk, so carry an empty candidate; the usage data
            # rides along in ``metadata`` and is folded into TURN_COMPLETE.
            final_response_chunk = LLMChatResponseChunk(
                result=LLMChatCandidateChunk(),
                metadata=overall_meta,
            )
            yield final_response_chunk


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
    candidates = []
    for choice in openai_response.choices:
        msg = choice.message
        # 2) Build tool_calls list if present
        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                try:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            type=tc.type,
                            function=FunctionCall(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Invalid tool_call entry {tc}: {e}")

        # 3) Build function_call if present
        # OpenAI's legacy ``function_call`` field is retained for compatibility
        fc = msg.function_call
        function_call = (
            FunctionCall(
                name=fc.name or "",
                arguments=fc.arguments or "",
            )
            if fc
            else None
        )

        # 4) Assemble AssistantMessage
        assistant_message = AssistantMessage(
            content=msg.content,
            refusal=msg.refusal,
            tool_calls=tool_calls,
            function_call=function_call,
        )

        # 5) Build candidate, including index & logprobs
        candidate = LLMChatCandidate(
            message=assistant_message,
            finish_reason=choice.finish_reason,
            index=choice.index,
            logprobs=choice.logprobs,
        )
        candidates.append(candidate)

    # 6) Metadata: include provider tag
    metadata: Dict[str, Any] = {
        "provider": "openai",
        "id": openai_response.id,
        "model": openai_response.model,
        "object": openai_response.object,
        "usage": openai_response.usage,
        "created": openai_response.created,
        "service_tier": openai_response.service_tier,
        "system_fingerprint": openai_response.system_fingerprint,
    }

    return LLMChatResponse(results=candidates, metadata=metadata)
