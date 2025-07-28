import dataclasses
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
    pkt: Dict[str, Any], enrich_metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract metadata from OpenAI packet and merge with enrich_metadata.

    Args:
        pkt (Dict[str, Any]): The OpenAI packet from which to extract metadata.
        enrich_metadata (Optional[Dict[str, Any]]): Additional metadata to merge with the extracted metadata.

    Returns:
        Dict[str, Any]: The merged metadata dictionary.
    """

    try:
        return {
            "id": pkt.get("id"),
            "created": pkt.get("created"),
            "model": pkt.get("model"),
            "object": pkt.get("object"),
            "service_tier": pkt.get("service_tier"),
            "system_fingerprint": pkt.get("system_fingerprint"),
            **(enrich_metadata or {}),
        }
    except Exception as e:
        logger.error(f"Failed to parse packet: {e}", exc_info=True)
        return {}


# Helper function to process each choice delta (content, function call, tool call, finish reason)
def _process_choice_delta(
    choice: Dict[str, Any],
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
    delta: dict = choice.get("delta", {})
    idx = choice.get("index")
    finish_reason = choice.get("finish_reason", None)
    logprobs = choice.get("logprobs", None)

    # Set additional metadata
    if finish_reason in ("stop", "tool_calls"):
        meta["last_chunk"] = True

    # Process content delta
    content = delta.get("content", None)
    function_call = delta.get("function_call", None)
    refusal = delta.get("refusal", None)
    role = delta.get("role", None)

    # Process tool calls
    chunk_tool_calls = [ToolCallChunk(**tc) for tc in (delta.get("tool_calls") or [])]

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
        # Convert Pydantic / OpenAIObject → plain dict
        if hasattr(packet, "model_dump"):
            pkt = packet.model_dump()
        elif hasattr(packet, "to_dict"):
            pkt = packet.to_dict()
        elif dataclasses.is_dataclass(packet):
            pkt = dataclasses.asdict(packet)
        else:
            raise TypeError(f"Cannot serialize packet of type {type(packet)}")

        # Capture overall metadata from the packet
        overall_meta = _get_packet_metadata(pkt, enrich_metadata)

        # Process each choice in this packet
        if choices := pkt.get("choices"):
            if len(choices) == 0:
                logger.warning("Received empty 'choices' in OpenAI packet, skipping.")
                continue
            # Process the first choice in the packet
            choice = choices[0]
            yield from _process_choice_delta(
                choice, overall_meta, on_chunk, first_chunk_flag
            )
            # Set first_chunk_flag to False after processing the first choice
            first_chunk_flag = False
        else:
            logger.warning(f" Yielding packet without 'choices': {pkt}")
            # Initialize default LLMChatResponseChunk
            final_response_chunk = LLMChatResponseChunk(metadata=overall_meta)
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
