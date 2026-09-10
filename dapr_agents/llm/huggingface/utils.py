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

from huggingface_hub import ChatCompletionOutput, ChatCompletionStreamOutput

from dapr_agents.types.message import (
    AssistantMessage,
    FunctionCall,
    LLMChatCandidate,
    LLMChatCandidateChunk,
    LLMChatResponse,
    LLMChatResponseChunk,
    ToolCall,
    ToolCallChunk,
)

logger = logging.getLogger(__name__)


# Helper function to handle metadata extraction
def _get_packet_metadata(
    packet: ChatCompletionStreamOutput, enrich_metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract metadata from HuggingFace packet and merge with enrich_metadata.

    Args:
        packet (ChatCompletionStreamOutput): The HuggingFace packet from which to extract metadata.
        enrich_metadata (Optional[Dict[str, Any]]): Additional metadata to merge with the extracted metadata.

    Returns:
        Dict[str, Any]: The merged metadata dictionary.
    """

    try:
        return {
            "id": packet.id,
            "created": packet.created,
            "model": packet.model,
            "object": None,
            "service_tier": None,
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
        choice (Dict[str, Any]): The choice delta from HuggingFace response.
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
    # HF exposes function calls through ``tool_calls`` and has no
    # refusal field in its typed stream delta; omit them and
    # fall back to LLMChatCandidateChunk's ``None`` defaults.
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


# Main function to process HuggingFace streaming response
def process_hf_stream(
    raw_stream: Iterator[ChatCompletionStreamOutput],
    *,
    enrich_metadata: Optional[Dict[str, Any]] = None,
    on_chunk: Optional[Callable],
) -> Iterator[LLMChatCandidateChunk]:
    """
    Normalize HuggingFace streaming chat into LLMChatCandidateChunk objects,
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
                "Yielding final packet without 'choices' (usage-only): %s", packet
            )
            # Final usage-only packet (empty ``choices``). ``result`` is required
            # on LLMChatResponseChunk, so carry an empty candidate; usage data
            # rides along in ``metadata`` and is folded into TURN_COMPLETE.
            final_response_chunk = LLMChatResponseChunk(
                result=LLMChatCandidateChunk(),
                metadata=overall_meta,
            )
            yield final_response_chunk


def process_hf_chat_response(response: ChatCompletionOutput) -> LLMChatResponse:
    """
    Convert a non-streaming Hugging Face ChatCompletionOutput into our unified LLMChatResponse.

    This will:
      1. Extract each typed `choice`, build an AssistantMessage (including any tool_calls or
         function_call shortcuts), wrap in LLMChatCandidate.
      3. Collect top-level metadata (id, model, usage, etc.) into an LLMChatResponse.

    Args:
        response: The HFHubInferenceClientBase.chat.completions.create(...) output.

    Returns:
        An LLMChatResponse containing all chat candidates and metadata.
    """
    candidates = []
    for choice in response.choices:
        msg = choice.message

        # 2) build tool_calls list if present
        tool_calls: Optional[list[ToolCall]] = None
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
                except Exception:
                    logger.exception(f"Invalid HF tool_call entry: {tc}")

        # 2b) handle the single‑ID shortcut
        if msg.tool_call_id and not tool_calls:
            # HF only sent you an ID; we turn that into a zero‑arg function_call
            fc = FunctionCall(name=msg.tool_call_id, arguments="{}")
            tool_calls = [ToolCall(id=msg.tool_call_id, type="function", function=fc)]

        # 3) promote first tool_call into function_call if desired
        function_call = tool_calls[0].function if tool_calls else None

        # HF has no refusal field; omit it
        assistant = AssistantMessage(
            content=msg.content,
            tool_calls=tool_calls,
            function_call=function_call,
        )

        candidates.append(
            LLMChatCandidate(
                message=assistant,
                finish_reason=choice.finish_reason,
                index=choice.index,
                logprobs=choice.logprobs,
            )
        )

    # 4) collect overall metadata
    metadata = {
        "provider": "huggingface",
        "id": response.id,
        "model": response.model,
        "created": response.created,
        "system_fingerprint": response.system_fingerprint,
        "usage": response.usage,
    }

    return LLMChatResponse(results=candidates, metadata=metadata)
