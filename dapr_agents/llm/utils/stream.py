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

from collections.abc import Mapping
import dataclasses
import logging
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    Optional,
    TypeVar,
)

from pydantic import BaseModel

from dapr_agents.llm.utils.providers import PROVIDERS_WITH_STREAMING

from dapr_agents.types.message import (
    FunctionCall,
    FunctionCallChunk,
    LLMChatCandidateChunk,
    LLMChatResponseChunk,
    ToolCallChunk,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _dump_obj(obj: Any) -> Dict[str, Any]:
    """Serialize SDK model, dataclass, or object to dictionary without silent failure."""
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return dict(obj)


def extract_packet_metadata(
    pkt: Any, enrich_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Extract standard completion metadata from packet and merge with enrich_metadata.

    Args:
        pkt (Any): The packet (dict or SDK object) from which to extract metadata.
        enrich_metadata (Optional[Dict[str, Any]]): Additional metadata to merge.

    Returns:
        Dict[str, Any]: The merged metadata dictionary.
    """
    meta: Dict[str, Any] = {}
    usage_raw = None
    try:
        if isinstance(pkt, Mapping):
            meta = {
                "id": pkt.get("id"),
                "created": pkt.get("created"),
                "model": pkt.get("model"),
                "object": pkt.get("object"),
                "service_tier": pkt.get("service_tier"),
                "system_fingerprint": pkt.get("system_fingerprint"),
            }
            usage_raw = pkt.get("usage")
        else:
            meta = {
                "id": getattr(pkt, "id", None),
                "created": getattr(pkt, "created", None),
                "model": getattr(pkt, "model", None),
                "object": getattr(pkt, "object", None),
                "service_tier": getattr(pkt, "service_tier", None),
                "system_fingerprint": getattr(pkt, "system_fingerprint", None),
            }
            usage_raw = getattr(pkt, "usage", None)
    except Exception as e:
        logger.warning(
            f"Failed to extract metadata fields from packet: {e}", exc_info=True
        )

    if usage_raw is not None:
        try:
            meta["usage"] = _dump_obj(usage_raw)
        except Exception as e:
            logger.warning(f"Failed to serialize usage object: {e}", exc_info=True)

    if enrich_metadata:
        meta.update(enrich_metadata)
    return meta


def process_choice_delta(
    choice: Any,
    overall_meta: Dict[str, Any],
    on_chunk: Optional[Callable] = None,
    first_chunk_flag: bool = False,
) -> Iterator[LLMChatResponseChunk]:
    """Process each choice delta and yield corresponding chunks.

    Args:
        choice (Any): The choice delta (dict or SDK object) from LLM response packet.
        overall_meta (Dict[str, Any]): Overall metadata to include in chunks.
        on_chunk (Optional[Callable]): Callback for each chunk.
        first_chunk_flag (bool): Flag indicating if this is the first choice-containing chunk in the stream.

    Yields:
        LLMChatResponseChunk: The processed chunk with content, function call, tool calls.
    """
    # Make an isolated snapshot (shallow-copied) for this single chunk
    meta = {**overall_meta}

    # mark first_chunk exactly once
    if first_chunk_flag and "first_chunk" not in meta:
        meta["first_chunk"] = True

    # Extract initial properties from choice
    if isinstance(choice, Mapping):
        delta = choice.get("delta") or {}
        idx = choice.get("index")
        finish_reason = choice.get("finish_reason", None)
        logprobs = choice.get("logprobs", None)
    else:
        delta = getattr(choice, "delta", None) or {}
        idx = getattr(choice, "index", None)
        finish_reason = getattr(choice, "finish_reason", None)
        logprobs = getattr(choice, "logprobs", None)

    # Set additional metadata
    if finish_reason is not None:
        meta["last_chunk"] = True

    # Process content delta
    if isinstance(delta, Mapping):
        content = delta.get("content", None)
        function_call = delta.get("function_call", None)
        refusal = delta.get("refusal", None)
        role = delta.get("role", None)
        tool_calls_raw = delta.get("tool_calls") or []
    else:
        content = getattr(delta, "content", None)
        function_call = getattr(delta, "function_call", None)
        refusal = getattr(delta, "refusal", None)
        role = getattr(delta, "role", None)
        tool_calls_raw = getattr(delta, "tool_calls", None) or []

    if function_call is not None and not isinstance(function_call, dict):
        function_call = {
            "name": getattr(function_call, "name", None),
            "arguments": getattr(function_call, "arguments", None),
        }

    # Process tool calls defensively
    chunk_tool_calls = []
    for tc in tool_calls_raw:
        try:
            if isinstance(tc, ToolCallChunk):
                chunk_tool_calls.append(tc)
            elif isinstance(tc, dict):
                chunk_tool_calls.append(ToolCallChunk(**tc))
            elif not isinstance(
                tc, (str, bytes, int, float, bool, list, tuple, set)
            ) and (hasattr(tc, "index") or hasattr(tc, "function")):
                fn_obj = getattr(tc, "function", None)
                if fn_obj is not None:
                    fn_chunk = (
                        FunctionCallChunk(**fn_obj)
                        if isinstance(fn_obj, dict)
                        else FunctionCallChunk(
                            name=getattr(fn_obj, "name", None),
                            arguments=getattr(fn_obj, "arguments", None),
                        )
                    )
                else:
                    fn_chunk = FunctionCallChunk()
                chunk_tool_calls.append(
                    ToolCallChunk(
                        index=getattr(tc, "index", 0),
                        id=getattr(tc, "id", None),
                        type=getattr(tc, "type", None),
                        function=fn_chunk,
                    )
                )
            else:
                raise TypeError(f"Invalid tool_call entry: {tc!r}")
        except Exception:
            logger.warning(f"Invalid tool_call entry in delta {tc}", exc_info=True)

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


@contextmanager
def managed_stream(stream: Any) -> Generator[Any, None, None]:
    """Normalize a stream-like object into a context manager.

    - If `stream` is already a context manager, use it as-is.
    - Otherwise, if it has `.close()`, call that on exit.
    - Otherwise, yield it unchanged with no cleanup.

    Cleanup failures are logged and then re-raised. If the body itself
    raised, the cleanup exception propagates with the original exception
    attached as `__context__`.
    """
    enter = getattr(stream, "__enter__", None)
    exit_ = getattr(stream, "__exit__", None)
    close = getattr(stream, "close", None)

    if callable(enter) and callable(exit_):
        with stream as s:
            yield s if s is not None else stream
    elif callable(close):
        try:
            yield stream
        finally:
            try:
                close()
            except Exception:
                logger.debug("Failed to close streaming response", exc_info=True)
                raise
    else:
        yield stream


def _extract_choices_and_metadata(
    packet: Any, enrich_metadata: Optional[Dict[str, Any]]
) -> tuple[Any, Dict[str, Any]]:
    """Extract choices iterable and overall metadata from packet (dict, SDK object, or wrapper)."""
    if isinstance(packet, Mapping):
        return packet.get("choices"), extract_packet_metadata(packet, enrich_metadata)

    choices = getattr(packet, "choices", None)
    if isinstance(choices, (list, tuple)):
        return choices, extract_packet_metadata(packet, enrich_metadata)

    # Fallback for wrapper objects that serialize via model_dump or to_dict
    if hasattr(packet, "model_dump") and callable(packet.model_dump):
        d = packet.model_dump()
        if isinstance(d, Mapping):
            return d.get("choices"), extract_packet_metadata(d, enrich_metadata)
    elif hasattr(packet, "to_dict") and callable(packet.to_dict):
        d = packet.to_dict()
        if isinstance(d, Mapping):
            return d.get("choices"), extract_packet_metadata(d, enrich_metadata)

    return None, extract_packet_metadata(packet, enrich_metadata)


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

    for packet in raw_stream:
        if isinstance(packet, (str, bytes, int, float, bool, list, set, tuple)):
            raise TypeError(f"Cannot serialize packet of type {type(packet)}")

        choices, overall_meta = _extract_choices_and_metadata(packet, enrich_metadata)

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
                f"Yielding final packet without 'choices' (usage-only): {packet}"
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

        Owns stream lifecycle and cleanup: ensures the underlying stream is
        closed or context-exited upon completion, error, or consumer early break.

        Args:
            stream:           Raw stream object or iterator from the provider SDK.
            llm_provider:     Name of the LLM provider (e.g., "openai", "anthropic").
            on_chunk:         Callback fired on every partial chunk.

        Yields:
            LLMChatResponseChunk: fully-typed chunks, partial and final.
        """
        provider = llm_provider.lower() if llm_provider else ""
        if provider not in PROVIDERS_WITH_STREAMING:
            raise ValueError(f"Streaming not supported for provider: {llm_provider}")

        with managed_stream(stream) as s:
            if provider in (
                "openai",
                "nvidia",
                "litellm",
                "iflytek",
                "huggingface",
            ):
                yield from process_choice_delta_stream(
                    raw_stream=s,
                    enrich_metadata={"provider": provider},
                    on_chunk=on_chunk,
                )
            elif provider in ("anthropic", "claude"):
                from dapr_agents.llm.anthropic.utils import process_anthropic_stream

                yield from process_anthropic_stream(
                    raw_stream=s,
                    enrich_metadata={"provider": provider},
                    on_chunk=on_chunk,
                )
            else:
                raise ValueError(
                    f"Streaming not supported for provider: {llm_provider}"
                )


__all__ = [
    "StreamHandler",
    "extract_packet_metadata",
    "managed_stream",
    "process_choice_delta",
    "process_choice_delta_stream",
]
