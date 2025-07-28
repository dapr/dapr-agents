import dataclasses
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
    pkt: Dict[str, Any], enrich_metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract metadata from HuggingFace packet and merge with enrich_metadata.

    Args:
        pkt (Dict[str, Any]): The HuggingFace packet from which to extract metadata.
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
        # Convert Pydantic / HuggingFaceObject → plain dict
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
                logger.warning(
                    "Received empty 'choices' in HuggingFace packet, skipping."
                )
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


def process_hf_chat_response(response: ChatCompletionOutput) -> LLMChatResponse:
    """
    Convert a non-streaming Hugging Face ChatCompletionOutput into our unified LLMChatResponse.

    This will:
      1. Turn the HF dataclass into a plain dict via .model_dump() or .dict().
      2. Extract each `choice`, build an AssistantMessage (including any tool_calls or
         function_call shortcuts), wrap in LLMChatCandidate.
      3. Collect top-level metadata (id, model, usage, etc.) into an LLMChatResponse.

    Args:
        response: The HFHubInferenceClientBase.chat.completions.create(...) output.

    Returns:
        An LLMChatResponse containing all chat candidates and metadata.
    """
    # 1) serialise the HF object to a primitive dict
    try:
        if hasattr(response, "model_dump"):
            resp: Dict[str, Any] = response.model_dump()
        elif hasattr(response, "dict"):
            resp: Dict[str, Any] = response.dict()
        elif dataclasses.is_dataclass(response):
            resp = dataclasses.asdict(response)
        elif hasattr(response, "to_dict"):
            resp = response.to_dict()
        else:
            raise TypeError(f"Cannot serialize object of type {type(response)}")
    except Exception:
        logger.exception("Failed to serialize HF chat response")
        resp = {}

    candidates = []
    for choice in resp.get("choices", []):
        msg = choice.get("message") or {}

        # 2) build tool_calls list if present
        tool_calls: Optional[list[ToolCall]] = None
        if msg.get("tool_calls"):
            tool_calls = []
            for tc in msg["tool_calls"]:
                try:
                    tool_calls.append(ToolCall(**tc))
                except Exception:
                    logger.exception(f"Invalid HF tool_call entry: {tc}")

        # 2b) handle the single‑ID shortcut
        if msg.get("tool_call_id") and not tool_calls:
            # HF only sent you an ID; we turn that into a zero‑arg function_call
            fc = FunctionCall(name=msg["tool_call_id"], arguments="")
            tool_calls = [
                ToolCall(id=msg["tool_call_id"], type="function", function=fc)
            ]

        # 3) promote first tool_call into function_call if desired
        function_call = tool_calls[0].function if tool_calls else None

        assistant = AssistantMessage(
            content=msg.get("content"),
            refusal=None,
            tool_calls=tool_calls,
            function_call=function_call,
        )

        candidates.append(
            LLMChatCandidate(
                message=assistant,
                finish_reason=choice.get("finish_reason"),
                index=choice.get("index"),
                logprobs=choice.get("logprobs"),
            )
        )

    # 4) collect overall metadata
    metadata = {
        "provider": "huggingface",
        "id": resp.get("id"),
        "model": resp.get("model"),
        "created": resp.get("created"),
        "system_fingerprint": resp.get("system_fingerprint"),
        "usage": resp.get("usage"),
    }

    return LLMChatResponse(results=candidates, metadata=metadata)
