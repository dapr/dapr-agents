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
from collections.abc import Iterator
from typing import Any

from anthropic import Anthropic
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
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content_str,
            }
        ],
    }


def as_assistant_with_tool_use(msg: dict[str, Any]) -> dict[str, Any]:
    """dapr-agents assistant with `tool_calls` -> Anthropic assistant with `tool_use` blocks"""
    blocks: list[dict[str, Any]] = []

    content = msg.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})

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
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_call_id,
                "name": function_name,
                "input": args_parsed,
            }
        )
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
    resp: Any, response_format: type[BaseModel]
) -> BaseModel | list[BaseModel]:
    """Read the forced `tool_use` block back into the Pydantic model"""
    target_format, target_model = resolve_target_model(response_format)
    for block in resp.content or []:
        if block.type == "tool_use" and block.name == target_model.__name__:
            return StructureHandler.validate_response(block.input, target_format)
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
    resp: Any, response_format: type[BaseModel]
) -> BaseModel | list[BaseModel]:
    """Validate the text-block JSON against the Pydantic model."""
    target_format, _ = resolve_target_model(response_format)
    for block in resp.content or []:
        if block.type == "text" and block.text:
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


def to_llm_chat_response(resp: Any) -> LLMChatResponse:
    """Note: Non-text/tool_use blocks (e.g. `thinking`) are stored in `metadata['raw_content']`"""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    raw_content: list[dict[str, Any]] = []

    for block in resp.content or []:
        raw_content.append(
            block.model_dump() if hasattr(block, "model_dump") else vars(block)
        )
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            function = FunctionCall(
                name=block.name,
                arguments=json.dumps(block.input or {}),
            )
            tool_calls.append(ToolCall(id=block.id, type="function", function=function))

    assistant = AssistantMessage(
        content="".join(text_parts) if text_parts else None,
        tool_calls=tool_calls or None,
    )
    usage = resp.usage.model_dump() if resp.usage is not None else None
    metadata: dict[str, Any] = {
        "provider": PROVIDER,
        "id": resp.id,
        "model": resp.model,
        "stop_reason": resp.stop_reason,
        "stop_sequence": resp.stop_sequence,
        "usage": usage,
        "raw_content": raw_content,
    }
    candidate = LLMChatCandidate(message=assistant, finish_reason=resp.stop_reason)
    return LLMChatResponse(results=[candidate], metadata=metadata)


def iter_stream(
    client: Anthropic, params: dict[str, Any]
) -> Iterator[LLMChatResponseChunk]:
    """Translate Anthropic SSE events into `LLMChatResponseChunk`s.

    Each yield gets a `dict(meta)` snapshot so buffered consumers don't all
    see the post-stream state.
    """
    meta: dict[str, Any] = {"provider": PROVIDER}

    try:
        with client.messages.create(stream=True, **params) as raw_stream:
            for event in raw_stream:
                if event.type == "message_start":
                    meta["id"] = event.message.id
                    meta["model"] = event.message.model

                elif event.type == "content_block_start":
                    # Text blocks only yield on the first `text_delta`; tool_use
                    # blocks must yield here because id/name only arrive on start.
                    block = event.content_block
                    if block.type == "tool_use":
                        function_chunk = FunctionCallChunk(
                            name=block.name, arguments=""
                        )
                        tool_call_chunk = ToolCallChunk(
                            index=event.index,
                            id=block.id,
                            type="function",
                            function=function_chunk,
                        )
                        candidate = LLMChatCandidateChunk(
                            role="assistant",
                            index=0,
                            tool_calls=[tool_call_chunk],
                        )
                        yield LLMChatResponseChunk(
                            result=candidate, metadata=dict(meta)
                        )

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        candidate = LLMChatCandidateChunk(
                            role="assistant",
                            content=delta.text,
                            index=0,
                        )
                        yield LLMChatResponseChunk(
                            result=candidate, metadata=dict(meta)
                        )
                    elif delta.type == "input_json_delta":
                        function_chunk = FunctionCallChunk(arguments=delta.partial_json)
                        tool_call_chunk = ToolCallChunk(
                            index=event.index, function=function_chunk
                        )
                        candidate = LLMChatCandidateChunk(
                            role="assistant",
                            index=0,
                            tool_calls=[tool_call_chunk],
                        )
                        yield LLMChatResponseChunk(
                            result=candidate, metadata=dict(meta)
                        )

                elif event.type == "message_delta":
                    if event.usage is not None:
                        meta["usage"] = event.usage.model_dump()
                    if event.delta.stop_reason:
                        candidate = LLMChatCandidateChunk(
                            finish_reason=event.delta.stop_reason
                        )
                        yield LLMChatResponseChunk(
                            result=candidate, metadata=dict(meta)
                        )
                # message_stop / content_block_stop / ping: ignored
    except Exception:
        logger.exception("Anthropic Messages API streaming call failed")
        raise
