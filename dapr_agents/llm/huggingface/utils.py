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

from huggingface_hub import ChatCompletionOutput, ChatCompletionStreamOutput

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


def process_hf_stream(
    raw_stream: Iterator[ChatCompletionStreamOutput],
    *,
    enrich_metadata: Optional[Dict[str, Any]] = None,
    on_chunk: Optional[Callable] = None,
) -> Iterator[LLMChatResponseChunk]:
    """Normalize HuggingFace streaming chat into LLMChatResponseChunk objects,
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


__all__ = [
    "process_hf_stream",
    "process_hf_chat_response",
    "_get_packet_metadata",
    "_process_choice_delta",
]
