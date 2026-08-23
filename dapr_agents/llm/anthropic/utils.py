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
import functools
import json
import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from anthropic import Anthropic, Stream
from anthropic.types import (
    Base64ImageSourceParam,
    ContentBlock,
    ImageBlockParam,
    InputJSONDelta,
    Message,
    MessageParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStreamEvent,
    RedactedThinkingBlock,
    SignatureDelta,
    TextBlock,
    TextBlockParam,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
    Usage,
)
from pydantic import BaseModel

from dapr_agents.llm.anthropic.client import PROVIDER
from dapr_agents.llm.utils import StructureHandler
from dapr_agents.tool.utils.function_calling import to_claude_function_call_definition
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


# ---------------------------------------------------------------------------
# dapr-agents message dicts -> Anthropic request format
# ---------------------------------------------------------------------------


def _normalize_content_blocks(content: Any) -> Any:
    """Translate OpenAI-style content blocks into Anthropic content blocks."""
    if not isinstance(content, list):
        return content

    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            text_block: TextBlockParam = {"type": "text", "text": item}
            blocks.append(text_block)
        elif isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "text":
                blocks.append(item)
            elif item_type == "image_url":
                image_url_dict = item.get("image_url", {})
                url = (
                    image_url_dict.get("url", "")
                    if isinstance(image_url_dict, dict)
                    else str(image_url_dict)
                )
                if url.startswith("data:"):
                    try:
                        header, base64_data = url.split(",", 1)
                        mime_type = header.split(";")[0].split(":")[1]
                        image_block: ImageBlockParam = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,  # type: ignore[typeddict-item]
                                "data": base64_data,
                            },
                        }
                        blocks.append(image_block)
                    except Exception as exc:
                        logger.warning(
                            f"Failed to parse base64 data URI in content block; "
                            f"passing block through unchanged: {exc}"
                        )
                        blocks.append(item)
                elif url.startswith(("http://", "https://")):
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url,
                            },
                        }
                    )
                else:
                    blocks.append(item)
            else:
                blocks.append(item)
        else:
            blocks.append(item)
    return blocks


def split_messages(
    normalized: list[dict[str, Any]],
) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Pull system messages into Anthropic's top-level `system` param.

    Returns block lists (not joined strings) when any system message arrived
    as blocks, so `cache_control` markers survive translation.
    """
    system_strs: list[str] = []
    system_blocks: list[dict[str, Any]] = []
    has_structured_system = False
    out: list[dict[str, Any]] = []

    for msg in normalized:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str) and content:
                system_strs.append(content)
                system_blocks.append({"type": "text", "text": content})
            elif isinstance(content, list) and content:
                has_structured_system = True
                system_blocks.extend(content)
        elif role == "tool":
            out.append(as_tool_result(msg))
        elif role == "assistant" and msg.get("tool_calls"):
            out.append(as_assistant_with_tool_use(msg))
        else:
            # Anthropic user/assistant turns only accept `role` and `content`.
            # Drop OpenAI-only keys (`name`, empty `tool_calls`, `function_call`,
            # `tool_call_id`, etc.) that would otherwise trigger request validation errors.
            anthropic_keys = ("role", "content")
            msg_clean = {k: v for k, v in msg.items() if k in anthropic_keys}
            if "content" in msg_clean:
                msg_clean["content"] = _normalize_content_blocks(msg_clean["content"])
            out.append(msg_clean)

    if has_structured_system:
        return system_blocks, out
    if system_strs:
        return "\n\n".join(system_strs), out
    return None, out


def as_tool_result(msg: dict[str, Any]) -> dict[str, Any]:
    """dapr-agents `{"role": "tool"}` -> Anthropic `tool_result` block on a user turn"""
    tool_call_id = msg.get("tool_call_id")
    if not tool_call_id:
        raise ValueError(
            "Cannot translate tool message to Anthropic: `tool_call_id` is required."
        )
    content = msg.get("content")
    content_str = content if isinstance(content, str) else json.dumps(content)
    tool_result_block: ToolResultBlockParam = {
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "content": content_str,
    }
    return {
        "role": "user",
        "content": [tool_result_block],
    }


def as_assistant_with_tool_use(msg: dict[str, Any]) -> dict[str, Any]:
    """dapr-agents assistant with `tool_calls` -> Anthropic assistant with `tool_use` blocks"""
    blocks: list[dict[str, Any]] = []

    content = msg.get("content")
    if isinstance(content, str) and content:
        text_block: TextBlockParam = {"type": "text", "text": content}
        blocks.append(text_block)

    for tool_call in msg["tool_calls"]:
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
        function_name = function.get("name")
        if not tool_call_id or not function_name:
            raise ValueError(
                f"Cannot translate tool_call to Anthropic: both `id` and `function.name` "
                f"are required (got id={tool_call_id!r}, name={function_name!r})."
            )
        args_raw = function.get("arguments", "{}")
        try:
            args_parsed = (
                json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Cannot translate tool_call {tool_call_id!r} ({function_name!r}) "
                f"to Anthropic: arguments are not valid JSON: {args_raw!r}"
            ) from exc
        if not isinstance(args_parsed, dict):
            raise ValueError(
                f"Cannot translate tool_call {tool_call_id!r} ({function_name!r}) "
                f"to Anthropic: arguments must decode to a JSON object, got {type(args_parsed).__name__}."
            )
        tool_use_block: ToolUseBlockParam = {
            "type": "tool_use",
            "id": tool_call_id,
            "name": function_name,
            "input": args_parsed,
        }
        blocks.append(tool_use_block)
    return {"role": "assistant", "content": blocks}


# ---------------------------------------------------------------------------
# Structured output dispatch
# ---------------------------------------------------------------------------


def resolve_target_model(
    response_format: type[BaseModel],
) -> tuple[Any, type[BaseModel]]:
    """Resolve `response_format`, wrapping `list[Model]` into a generated `IterableModel`.

    Returns:
        The target format for validation, and the target_model for introspection (name, docstring and JSON schema)
    """
    target_format = StructureHandler.normalize_iterable_format(response_format)
    target_model = StructureHandler.resolve_response_model(target_format)
    if target_model is None:
        raise TypeError(
            f"response_format must resolve to a Pydantic model; got {response_format!r}"
        )
    return target_format, target_model


def inject_function_call_request(
    params: dict[str, Any], response_format: type[BaseModel]
) -> None:
    """Force a single tool call (structured output fallback for older models)."""
    _, target_model = resolve_target_model(response_format)
    tool = to_claude_function_call_definition(
        target_model.__name__,
        target_model.__doc__ or "",
        target_model,
    )
    tools_existing = params.get("tools") or []
    params["tools"] = tools_existing + [tool]
    params["tool_choice"] = {"type": "tool", "name": target_model.__name__}


def parse_function_call_response(
    resp: Message | Any, response_format: type[BaseModel]
) -> BaseModel | list[BaseModel]:
    """Read the forced `tool_use` block back into the Pydantic model"""
    target_format, target_model = resolve_target_model(response_format)
    content = getattr(resp, "content", None) or []
    for block in content:
        if (
            isinstance(block, ToolUseBlock)
            or getattr(block, "type", None) == "tool_use"
        ) and getattr(block, "name", None) == target_model.__name__:
            block_input = getattr(block, "input", None)
            return StructureHandler.validate_response(block_input, target_format)
    raise ValueError(
        f"No tool_use block for {target_model.__name__!r} in Anthropic response."
    )


@functools.cache
def _model_supports_json_output(client: Anthropic, model: str) -> bool:
    capabilities = client.models.retrieve(model).capabilities
    return bool(capabilities and capabilities.structured_outputs.supported)


def assert_json_output_supported(client: Anthropic, model: str) -> None:
    """Fetches the capabilities straight from the Anthropic SDK, and
    raises if `model` does not support JSON output.

    No-op if the fetch fails."""
    try:
        supported = _model_supports_json_output(client, model)
    except Exception:
        logger.warning(
            f"Could not verify structured_outputs capability for {model!r}; "
            "allowing request to proceed.",
            exc_info=True,
        )
        return

    if not supported:
        raise ValueError(
            f"Model {model!r} does not support structured_mode='json'. "
            "Pass structured_mode='function_call' instead."
        )


def inject_json_request(
    params: dict[str, Any], response_format: type[BaseModel]
) -> None:
    """Use Anthropic's native `output_config` (grammar-constrained JSON).

    Requires Sonnet 4.5+, Opus 4.1+, or Haiku 4.5+; older models must use
    `function_call` mode
    """
    _, target_model = resolve_target_model(response_format)
    strict_schema = StructureHandler.enforce_strict_json_schema(
        target_model.model_json_schema()
    )
    params["output_config"] = {
        "format": {"type": "json_schema", "schema": strict_schema}
    }


def parse_json_response(
    resp: Message | Any, response_format: type[BaseModel]
) -> BaseModel | list[BaseModel]:
    """Validate the text-block JSON against the Pydantic model."""
    target_format, _ = resolve_target_model(response_format)
    content = getattr(resp, "content", None) or []
    for block in content:
        if (
            isinstance(block, TextBlock) or getattr(block, "type", None) == "text"
        ) and getattr(block, "text", None):
            return StructureHandler.validate_response(block.text, target_format)
    raise ValueError("No text block carrying structured JSON in Anthropic response.")


STRUCTURED_INJECTORS = {
    "json": inject_json_request,
    "function_call": inject_function_call_request,
}
STRUCTURED_PARSERS = {
    "json": parse_json_response,
    "function_call": parse_function_call_response,
}


# ---------------------------------------------------------------------------
# Anthropic response format -> dapr-agents response types
# ---------------------------------------------------------------------------


def to_llm_chat_response(resp: Message | Any) -> LLMChatResponse:
    """Translate an Anthropic Message or SDK-compatible response to LLMChatResponse.

    Note: Non-text/tool_use blocks (e.g. `thinking`) are stored in `metadata['raw_content']`.
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    raw_content: list[dict[str, Any]] = []

    content = getattr(resp, "content", None) or []
    for block in content:
        raw_content.append(
            block.model_dump() if hasattr(block, "model_dump") else vars(block)
        )
        if isinstance(block, TextBlock) or getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                text_parts.append(text)
        elif (
            isinstance(block, ToolUseBlock)
            or getattr(block, "type", None) == "tool_use"
        ):
            block_input = getattr(block, "input", {}) or {}
            args_str = (
                json.dumps(block_input)
                if isinstance(block_input, (dict, list))
                else str(block_input)
            )
            function = FunctionCall(
                name=getattr(block, "name", ""),
                arguments=args_str,
            )
            tool_calls.append(
                ToolCall(
                    id=getattr(block, "id", ""),
                    type="function",
                    function=function,
                )
            )

    assistant = AssistantMessage(
        content="".join(text_parts) if text_parts else None,
        tool_calls=tool_calls or None,
    )
    usage_obj = getattr(resp, "usage", None)
    usage = (
        usage_obj.model_dump()
        if hasattr(usage_obj, "model_dump")
        else (vars(usage_obj) if usage_obj is not None else None)
    )
    metadata: dict[str, Any] = {
        "provider": PROVIDER,
        "id": getattr(resp, "id", None),
        "model": getattr(resp, "model", None),
        "stop_reason": getattr(resp, "stop_reason", None),
        "stop_sequence": getattr(resp, "stop_sequence", None),
        "usage": usage,
        "raw_content": raw_content,
    }
    candidate = LLMChatCandidate(
        message=assistant, finish_reason=getattr(resp, "stop_reason", None)
    )
    return LLMChatResponse(results=[candidate], metadata=metadata)


def _snapshot_stream_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Create an isolated snapshot of stream metadata, making copies of mutable containers."""
    snapshot = dict(meta)
    if "thinking_blocks" in snapshot:
        snapshot["thinking_blocks"] = list(snapshot["thinking_blocks"])
    if "thinking_deltas" in snapshot:
        snapshot["thinking_deltas"] = list(snapshot["thinking_deltas"])
    return snapshot


def process_anthropic_stream(
    raw_stream: Iterable[RawMessageStreamEvent] | Stream[RawMessageStreamEvent] | Any,
    *,
    enrich_metadata: dict[str, Any] | None = None,
    on_chunk: Callable[[LLMChatResponseChunk], None] | None = None,
) -> Iterator[LLMChatResponseChunk]:
    """Translate Anthropic SSE events into `LLMChatResponseChunk`s.

    Each yield gets an isolated snapshot so buffered consumers don't all
    see the post-stream state.
    """
    meta: dict[str, Any] = {**(enrich_metadata or {"provider": PROVIDER})}

    stream_iter = (
        raw_stream.__enter__() if hasattr(raw_stream, "__enter__") else raw_stream
    )
    try:
        for event in stream_iter:
            event_type = getattr(event, "type", None)

            if isinstance(event, RawMessageStartEvent) or event_type == "message_start":
                msg = getattr(event, "message", None)
                if msg:
                    meta["id"] = getattr(msg, "id", None)
                    meta["model"] = getattr(msg, "model", None)
                    usage = getattr(msg, "usage", None)
                    if usage is not None:
                        meta["usage"] = (
                            usage.model_dump()
                            if hasattr(usage, "model_dump")
                            else vars(usage)
                        )

            elif (
                isinstance(event, RawContentBlockStartEvent)
                or event_type == "content_block_start"
            ):
                block = getattr(event, "content_block", None)
                block_type = getattr(block, "type", None)
                idx = getattr(event, "index", 0)

                if isinstance(block, ToolUseBlock) or block_type == "tool_use":
                    function_chunk = FunctionCallChunk(
                        name=getattr(block, "name", ""), arguments=""
                    )
                    tool_call_chunk = ToolCallChunk(
                        index=idx,
                        id=getattr(block, "id", None),
                        type="function",
                        function=function_chunk,
                    )
                    candidate = LLMChatCandidateChunk(
                        role="assistant",
                        index=0,
                        tool_calls=[tool_call_chunk],
                    )
                    chunk = LLMChatResponseChunk(
                        result=candidate, metadata=_snapshot_stream_meta(meta)
                    )
                    if on_chunk:
                        on_chunk(chunk)
                    yield chunk
                elif isinstance(
                    block, (ThinkingBlock, RedactedThinkingBlock)
                ) or block_type in ("thinking", "redacted_thinking"):
                    thinking_text = getattr(block, "thinking", None) or getattr(
                        block, "data", ""
                    )
                    meta.setdefault("thinking_blocks", []).append(thinking_text)

            elif (
                isinstance(event, RawContentBlockDeltaEvent)
                or event_type == "content_block_delta"
            ):
                delta = getattr(event, "delta", None)
                delta_type = getattr(delta, "type", None)
                idx = getattr(event, "index", 0)

                if isinstance(delta, TextDelta) or delta_type == "text_delta":
                    candidate = LLMChatCandidateChunk(
                        role="assistant",
                        content=getattr(delta, "text", ""),
                        index=0,
                    )
                    chunk = LLMChatResponseChunk(
                        result=candidate, metadata=_snapshot_stream_meta(meta)
                    )
                    if on_chunk:
                        on_chunk(chunk)
                    yield chunk
                elif (
                    isinstance(delta, InputJSONDelta)
                    or delta_type == "input_json_delta"
                ):
                    function_chunk = FunctionCallChunk(
                        arguments=getattr(delta, "partial_json", "")
                    )
                    tool_call_chunk = ToolCallChunk(index=idx, function=function_chunk)
                    candidate = LLMChatCandidateChunk(
                        role="assistant",
                        index=0,
                        tool_calls=[tool_call_chunk],
                    )
                    chunk = LLMChatResponseChunk(
                        result=candidate, metadata=_snapshot_stream_meta(meta)
                    )
                    if on_chunk:
                        on_chunk(chunk)
                    yield chunk
                elif isinstance(delta, ThinkingDelta) or delta_type == "thinking_delta":
                    meta.setdefault("thinking_deltas", []).append(
                        getattr(delta, "thinking", "")
                    )
                elif (
                    isinstance(delta, SignatureDelta) or delta_type == "signature_delta"
                ):
                    meta["thinking_signature"] = getattr(delta, "signature", None)

            elif (
                isinstance(event, RawMessageDeltaEvent) or event_type == "message_delta"
            ):
                usage = getattr(event, "usage", None)
                if usage is not None:
                    meta["usage"] = (
                        usage.model_dump()
                        if hasattr(usage, "model_dump")
                        else vars(usage)
                    )
                delta = getattr(event, "delta", None)
                stop_reason = getattr(delta, "stop_reason", None)
                if stop_reason:
                    candidate = LLMChatCandidateChunk(finish_reason=stop_reason)
                    chunk = LLMChatResponseChunk(
                        result=candidate, metadata=_snapshot_stream_meta(meta)
                    )
                    if on_chunk:
                        on_chunk(chunk)
                    yield chunk
            # message_stop / content_block_stop / ping: ignored
    finally:
        if hasattr(raw_stream, "__exit__"):
            raw_stream.__exit__(None, None, None)


def iter_stream(
    client: Anthropic,
    params: dict[str, Any],
    on_chunk: Callable[[LLMChatResponseChunk], None] | None = None,
) -> Iterator[LLMChatResponseChunk]:
    """Translate Anthropic SSE events into `LLMChatResponseChunk`s."""
    try:
        raw_stream = client.messages.create(stream=True, **params)
        yield from process_anthropic_stream(
            raw_stream,
            enrich_metadata={"provider": PROVIDER},
            on_chunk=on_chunk,
        )
    except Exception:
        logger.exception("Anthropic Messages API streaming call failed")
        raise
