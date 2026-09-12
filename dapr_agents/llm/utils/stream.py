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
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    TypeVar,
)

from pydantic import BaseModel

from dapr_agents.types.message import (
    LLMChatCandidateChunk,
    LLMChatResponseChunk,
    ToolCallChunk,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_packet_metadata(
    pkt: Dict[str, Any], enrich_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Extract standard completion metadata from packet and merge with enrich_metadata.

    Args:
        pkt (Dict[str, Any]): The packet from which to extract metadata.
        enrich_metadata (Optional[Dict[str, Any]]): Additional metadata to merge.

    Returns:
        Dict[str, Any]: The merged metadata dictionary.
    """
    try:
        meta: Dict[str, Any] = {
            "id": pkt.get("id"),
            "created": pkt.get("created"),
            "model": pkt.get("model"),
            "object": pkt.get("object"),
            "service_tier": pkt.get("service_tier"),
            "system_fingerprint": pkt.get("system_fingerprint"),
        }
        if "usage" in pkt and pkt["usage"] is not None:
            meta["usage"] = pkt["usage"]
        if enrich_metadata:
            meta.update(enrich_metadata)
        return meta
    except Exception as e:
        logger.error(f"Failed to parse packet: {e}", exc_info=True)
        return {}


# Backward-compatible alias
_get_packet_metadata = extract_packet_metadata


def process_choice_delta(
    choice: Dict[str, Any],
    overall_meta: Dict[str, Any],
    on_chunk: Optional[Callable] = None,
    first_chunk_flag: bool = False,
) -> Iterator[LLMChatResponseChunk]:
    """Process each choice delta and yield corresponding chunks.

    Args:
        choice (Dict[str, Any]): The choice delta from LLM response packet.
        overall_meta (Dict[str, Any]): Overall metadata to include in chunks.
        on_chunk (Optional[Callable]): Callback for each chunk.
        first_chunk_flag (bool): Flag indicating if this is the first chunk.

    Yields:
        LLMChatResponseChunk: The processed chunk with content, function call, tool calls.
    """
    # Make an immutable snapshot for this single chunk
    meta = {**overall_meta}

    # mark first_chunk exactly once
    if first_chunk_flag and "first_chunk" not in meta:
        meta["first_chunk"] = True

    # Extract initial properties from choice
    choice_dict = choice or {}
    delta: dict = choice_dict.get("delta") or {}
    if not isinstance(delta, dict):
        delta = {}
    idx = choice_dict.get("index")
    finish_reason = choice_dict.get("finish_reason", None)
    logprobs = choice_dict.get("logprobs", None)

    # Set additional metadata
    if finish_reason in ("stop", "tool_calls"):
        meta["last_chunk"] = True

    # Process content delta
    content = delta.get("content", None)
    function_call = delta.get("function_call", None)
    refusal = delta.get("refusal", None)
    role = delta.get("role", None)

    # Process tool calls defensively
    chunk_tool_calls = []
    for tc in delta.get("tool_calls") or []:
        try:
            chunk_tool_calls.append(ToolCallChunk(**tc))
        except Exception as e:
            logger.warning(f"Invalid tool_call entry in delta {tc}: {e}")

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


# Backward-compatible alias
_process_choice_delta = process_choice_delta


def process_choice_delta_stream(
    raw_stream: Iterable[Any],
    *,
    enrich_metadata: Optional[Dict[str, Any]] = None,
    on_chunk: Optional[Callable] = None,
) -> Iterator[LLMChatResponseChunk]:
    """Normalize streaming chat completion responses following the choice-delta format.

    Used by OpenAI, Hugging Face, Nvidia, LiteLLM, and iFlytek to turn SDK chunk packets
    into standardized LLMChatResponseChunk objects, accumulating buffers per choice and
    yielding both partial and final chunks.

    Args:
        raw_stream: Raw iterator of chunk packets from the provider SDK.
        enrich_metadata: Extra key/value pairs to merge into each chunk.metadata.
        on_chunk: Callback fired on every partial delta (token, function, tool, usage).

    Yields:
        LLMChatResponseChunk for every partial and final piece, in stream order.
    """
    enrich_metadata = enrich_metadata or {}
    first_chunk_flag = True

    try:
        for packet in raw_stream:
            # Convert Pydantic / SDK object -> plain dict
            if hasattr(packet, "model_dump"):
                pkt = packet.model_dump()
            elif hasattr(packet, "to_dict"):
                pkt = packet.to_dict()
            elif dataclasses.is_dataclass(packet):
                pkt = dataclasses.asdict(packet)
            else:
                raise TypeError(f"Cannot serialize packet of type {type(packet)}")

            overall_meta = extract_packet_metadata(pkt, enrich_metadata)

            choices = pkt.get("choices")
            if choices:
                for choice in choices:
                    if not choice:
                        continue
                    yield from process_choice_delta(
                        choice, overall_meta, on_chunk, first_chunk_flag
                    )
                    first_chunk_flag = False
            else:
                logger.debug(
                    "Yielding final packet without 'choices' (usage-only): %s", pkt
                )
                # Final usage-only packet (empty ``choices``) sent when usage reporting
                # is enabled. ``result`` is required on LLMChatResponseChunk, so carry
                # an empty candidate; usage data rides along in ``metadata`` and is
                # folded into TURN_COMPLETE.
                final_response_chunk = LLMChatResponseChunk(
                    result=LLMChatCandidateChunk(),
                    metadata={**overall_meta},
                )
                if on_chunk:
                    on_chunk(final_response_chunk)
                yield final_response_chunk
    finally:
        if hasattr(raw_stream, "__exit__"):
            raw_stream.__exit__(None, None, None)
        elif hasattr(raw_stream, "close"):
            try:
                raw_stream.close()
            except Exception:
                pass


class StreamHandler:
    """Handles streaming of chat completion responses, delegating to the
    provider-specific stream processor and optionally validating output
    against Pydantic models.
    """

    @staticmethod
    def process_stream(
        stream: Iterable[Any],
        llm_provider: str,
        on_chunk: Optional[Callable] = None,
    ) -> Iterator[LLMChatResponseChunk]:
        """Process a streaming chat completion.

        Args:
            stream:           Raw stream object or iterator from the provider SDK.
            llm_provider:     Name of the LLM provider (e.g., "openai", "anthropic").
            on_chunk:         Callback fired on every partial chunk.

        Yields:
            LLMChatResponseChunk: fully-typed chunks, partial and final.
        """
        provider = llm_provider.lower() if llm_provider else ""
        if provider in (
            "openai",
            "nvidia",
            "litellm",
            "iflytek",
            "huggingface",
        ):
            yield from process_choice_delta_stream(
                raw_stream=stream,
                enrich_metadata={"provider": provider},
                on_chunk=on_chunk,
            )
        elif provider in ("anthropic", "claude"):
            from dapr_agents.llm.anthropic.utils import process_anthropic_stream

            yield from process_anthropic_stream(
                raw_stream=stream,
                enrich_metadata={"provider": provider},
                on_chunk=on_chunk,
            )
        else:
            raise ValueError(f"Streaming not supported for provider: {llm_provider}")


__all__ = [
    "StreamHandler",
    "extract_packet_metadata",
    "process_choice_delta",
    "process_choice_delta_stream",
    "_get_packet_metadata",
    "_process_choice_delta",
]
