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

"""Tests for the Claude SDK message -> ``AgentEvent`` mapping."""

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ServerToolResultBlock,
    ServerToolUseBlock,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk.types import DeferredToolUse  # noqa: E402

from dapr_agents.agents.executors.claude_events import (  # noqa: E402
    ClaudeEventMapper,
    tool_result_text,
)
from tests.executors.claude_fakes import SESSION_ID, result_message  # noqa: E402


def _display(name):
    prefix = "mcp__dapr__"
    if name.startswith(prefix):
        return name[len(prefix) :], "local"
    return name, "claude"


def _mapper(**kwargs):
    defaults = dict(
        session_id="initial",
        display_name=_display,
        approval_for=lambda call_id: {"reason": f"approve {call_id}"},
        include_text_deltas=True,
    )
    defaults.update(kwargs)
    return ClaudeEventMapper(**defaults)


def _text_delta(text, **kwargs):
    return StreamEvent(
        uuid="u",
        session_id=SESSION_ID,
        event={
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
        **kwargs,
    )


def _types(events):
    return [e.type for e in events]


class TestToolResultText:
    @pytest.mark.parametrize(
        "content, expected",
        [
            (None, ""),
            ("plain", "plain"),
            ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "a\nb"),
            ([{"type": "image", "data": "x"}], '[{"type": "image", "data": "x"}]'),
            ({"k": 1}, '{"k": 1}'),
        ],
    )
    def test_flattens(self, content, expected):
        assert tool_result_text(content) == expected


class TestStreamMapping:
    def test_init_emits_session_and_adopts_id(self):
        mapper = _mapper()
        events = mapper.map(
            SystemMessage(subtype="init", data={"session_id": SESSION_ID})
        )
        assert _types(events) == ["session"]
        assert events[0].session_id == SESSION_ID
        assert events[0].content == {"session_id": SESSION_ID, "stage": "init"}

    def test_other_system_subtypes_are_ignored(self):
        assert _mapper().map(SystemMessage(subtype="status", data={})) == []

    def test_text_delta(self):
        events = _mapper().map(_text_delta("Hi"))
        assert [(e.type, e.content) for e in events] == [("text_delta", "Hi")]

    @pytest.mark.parametrize(
        "message",
        [
            _text_delta("Hi", parent_tool_use_id="sub"),
            _text_delta(""),
            StreamEvent(
                uuid="u",
                session_id=SESSION_ID,
                event={
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "hmm"},
                },
            ),
            StreamEvent(
                uuid="u", session_id=SESSION_ID, event={"type": "message_start"}
            ),
        ],
    )
    def test_non_text_deltas_are_skipped(self, message):
        assert _mapper().map(message) == []

    def test_text_deltas_can_be_disabled(self):
        assert _mapper(include_text_deltas=False).map(_text_delta("Hi")) == []

    def test_unknown_messages_are_ignored(self):
        assert _mapper().map(object()) == []

    def test_assistant_text_and_tool_use(self):
        mapper = _mapper()
        events = mapper.map(
            AssistantMessage(
                content=[
                    ThinkingBlock(thinking="", signature="s"),
                    TextBlock(text="Checking"),
                    ToolUseBlock(
                        id="t1", name="mcp__dapr__weather", input={"city": "P"}
                    ),
                ],
                model="claude-x",
                message_id="m1",
                session_id=SESSION_ID,
            )
        )
        assert _types(events) == ["message", "tool_call"]
        assert events[0].content == {"role": "assistant", "content": "Checking"}
        assert events[0].metadata == {"message_id": "m1", "model": "claude-x"}
        assert events[1].content == {
            "id": "t1",
            "name": "weather",
            "arguments": {"city": "P"},
        }
        assert events[1].metadata == {"source": "local"}

    def test_tool_result_is_matched_to_call_name(self):
        mapper = _mapper()
        mapper.map(
            AssistantMessage(
                content=[ToolUseBlock(id="t1", name="mcp__dapr__weather", input={})],
                model="m",
            )
        )
        events = mapper.map(
            UserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="t1",
                        content=[{"type": "text", "text": "sunny"}],
                        is_error=None,
                    ),
                    TextBlock(text="ignored"),
                ]
            )
        )
        assert [e.content for e in events] == [
            {
                "tool_call_id": "t1",
                "name": "weather",
                "result": "sunny",
                "is_error": False,
            }
        ]

    def test_denied_tool_result_is_error(self):
        events = _mapper().map(
            UserMessage(
                content=[
                    ToolResultBlock(tool_use_id="t9", content="denied", is_error=True)
                ]
            )
        )
        assert events[0].content["is_error"] is True
        assert events[0].content["name"] == ""

    def test_server_tool_blocks(self):
        events = _mapper().map(
            AssistantMessage(
                content=[
                    ServerToolUseBlock(id="s1", name="web_search", input={"q": "x"}),
                    ServerToolResultBlock(tool_use_id="s1", content={"hits": 1}),
                ],
                model="m",
            )
        )
        assert _types(events) == ["tool_call", "tool_result"]
        assert events[1].content["name"] == "web_search"

    @pytest.mark.parametrize(
        "message",
        [
            AssistantMessage(
                content=[TextBlock(text="sub")], model="m", parent_tool_use_id="p"
            ),
            UserMessage(content="plain prompt"),
            UserMessage(
                content=[ToolResultBlock(tool_use_id="t", content="x")],
                parent_tool_use_id="p",
            ),
        ],
    )
    def test_subagent_and_prompt_messages_are_skipped(self, message):
        assert _mapper().map(message) == []


class TestResultMapping:
    def test_complete_with_usage_and_cost_delta(self):
        mapper = _mapper(prior_cost_usd=0.004)
        events = mapper.map(
            result_message(total_cost_usd=0.01, model_usage={"claude-x": {}})
        )
        assert _types(events) == ["session"]
        terminal = mapper.terminal
        assert terminal.type == "complete"
        assert terminal.content == {"role": "assistant", "content": "done"}
        assert terminal.session_id == SESSION_ID
        assert terminal.metadata["cost_usd"] == pytest.approx(0.006)
        assert terminal.metadata["session_total_cost_usd"] == 0.01
        assert terminal.metadata["usage"] == {"input_tokens": 3, "output_tokens": 2}
        assert terminal.metadata["model_usage"] == {"claude-x": {}}
        assert terminal.metadata["stop_reason"] == "end_turn"

    def test_cost_delta_never_negative_and_none_without_total(self):
        mapper = _mapper(prior_cost_usd=1.0)
        mapper.map(result_message(total_cost_usd=0.5))
        assert mapper.terminal.metadata["cost_usd"] == 0.0
        mapper = _mapper()
        mapper.map(result_message(total_cost_usd=None))
        assert mapper.terminal.metadata["cost_usd"] is None

    def test_complete_falls_back_to_last_text(self):
        mapper = _mapper()
        mapper.map(AssistantMessage(content=[TextBlock(text="final")], model="m"))
        mapper.map(result_message(result=None))
        assert mapper.terminal.content["content"] == "final"

    def test_deferred_tool_use_pauses(self):
        mapper = _mapper()
        mapper.map(
            result_message(
                result="",
                stop_reason="tool_deferred",
                deferred_tool_use=DeferredToolUse(
                    id="t1", name="mcp__dapr__pay", input={"to": "a"}
                ),
            )
        )
        terminal = mapper.terminal
        assert terminal.type == "paused"
        assert terminal.content == {
            "tool_call_id": "t1",
            "name": "pay",
            "arguments": {"to": "a"},
            "approval": {"reason": "approve t1"},
            "source": "local",
        }
        assert terminal.metadata["stop_reason"] == "tool_deferred"

    def test_error_result(self):
        mapper = _mapper()
        mapper.map(
            result_message(
                is_error=True,
                subtype="error_max_turns",
                terminal_reason="max_turns",
                errors=["Reached maximum number of turns (1)"],
                api_error_status=None,
            )
        )
        terminal = mapper.terminal
        assert terminal.type == "error"
        assert "subtype=error_max_turns" in terminal.content
        assert "Reached maximum number of turns" in terminal.content
        assert terminal.metadata["errors"] == ["Reached maximum number of turns (1)"]
        assert terminal.metadata["terminal_reason"] == "max_turns"

    def test_error_result_without_errors_uses_result_text(self):
        mapper = _mapper()
        mapper.map(result_message(is_error=True, result="Not logged in"))
        assert mapper.terminal.content.endswith("Not logged in")

    def test_error_helper(self):
        event = _mapper().error("boom", hint="x")
        assert (event.type, event.content, event.metadata) == (
            "error",
            "boom",
            {"hint": "x"},
        )
