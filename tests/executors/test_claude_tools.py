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

"""Tests for the dapr-agents tool bridge and the PreToolUse approval gate."""

import pytest

pytest.importorskip("claude_agent_sdk")

from dapr_agents.agents.executors import ToolCallDecision, arguments_digest  # noqa: E402
from dapr_agents.agents.executors.claude_tools import (  # noqa: E402
    PendingApproval,
    ToolGate,
    _to_sdk_tool,
    build_tool_server,
)
from dapr_agents.hooks import (  # noqa: E402
    Deny,
    Mutate,
    Proceed,
    RequireApproval,
    Skip,
)
from dapr_agents.tool import AgentTool, tool  # noqa: E402

PREFIX = "mcp__dapr__"


@tool
def get_weather(city: str) -> str:
    """Get weather."""
    return f"sunny in {city}"


@tool
def forecast(days: int) -> dict:
    """Structured result."""
    return {"days": days}


def _broken() -> str:
    raise RuntimeError("tool exploded")


broken = AgentTool(name="broken", description="Always fails.", func=_broken)


def _decision(output):
    return output["hookSpecificOutput"]["permissionDecision"]


def _input(tool_name, **tool_input):
    return {"tool_name": tool_name, "tool_input": tool_input}


class TestToolServer:
    def test_build_tool_server_registers_sdk_server(self):
        server = build_tool_server("dapr", [get_weather])
        assert server["type"] == "sdk"
        assert server["name"] == "dapr"

    def test_schema_comes_from_args_model(self):
        sdk_tool = _to_sdk_tool(get_weather)
        assert sdk_tool.name == get_weather.name
        assert "city" in sdk_tool.input_schema["properties"]

    def test_tool_without_args_model_gets_empty_schema(self):
        no_args = AgentTool(name="ping", description="Ping.", func=lambda: "pong")
        no_args.args_model = None
        assert _to_sdk_tool(no_args).input_schema == {
            "type": "object",
            "properties": {},
        }

    async def test_handler_returns_text(self):
        out = await _to_sdk_tool(get_weather).handler({"city": "Paris"})
        assert out == {"content": [{"type": "text", "text": "sunny in Paris"}]}

    async def test_handler_serializes_structured_results(self):
        out = await _to_sdk_tool(forecast).handler({"days": 2})
        assert out["content"][0]["text"] == '{"days": 2}'

    async def test_handler_returns_recorded_skip_result(self):
        def skipped(name, args):
            return "cached" if (name, args) == ("broken", {}) else None

        out = await _to_sdk_tool(broken, skipped).handler({})
        assert out == {"content": [{"type": "text", "text": "cached"}]}

    async def test_handler_reports_errors_to_the_model(self):
        out = await _to_sdk_tool(broken).handler({})
        assert out["is_error"] is True
        assert "tool exploded" in out["content"][0]["text"]


class TestToolGate:
    def _gate(self, *hooks, decisions=None, sources=None):
        return ToolGate(
            hooks=hooks,
            decisions=decisions or {},
            tool_prefix=PREFIX,
            tool_sources=sources,
        )

    def test_enabled_only_with_hooks_or_decisions(self):
        assert not self._gate().enabled
        assert self._gate(lambda c: None).enabled
        assert self._gate(decisions={"t": ToolCallDecision("t", True)}).enabled

    def test_display_name(self):
        gate = self._gate()
        assert gate.display_name(f"{PREFIX}pay") == ("pay", "local")
        assert gate.display_name("mcp__github__issue") == ("mcp__github__issue", "mcp")
        assert gate.display_name("Bash") == ("Bash", "claude")

    def test_bound_tools_report_their_own_source(self):
        gate = self._gate(sources={"delete_repo": "mcp"})
        assert gate.display_name(f"{PREFIX}delete_repo") == ("delete_repo", "mcp")
        assert gate.display_name(f"{PREFIX}pay") == ("pay", "local")

    async def test_source_keyed_hook_gates_bound_mcp_tool(self):
        def gate_mcp_deletes(ctx):
            if ctx.source == "mcp" and ctx.step_name.startswith("delete_"):
                return RequireApproval()
            return Proceed()

        gate = self._gate(gate_mcp_deletes, sources={"delete_repo": "mcp"})
        out = await gate(_input(f"{PREFIX}delete_repo"), "t1", None)
        assert _decision(out) == "defer"

    async def test_bound_mcp_tool_is_allowed_when_hooks_proceed(self):
        gate = self._gate(lambda c: Proceed(), sources={"lookup": "mcp"})
        out = await gate(_input(f"{PREFIX}lookup"), "t1", None)
        assert _decision(out) == "allow"

    async def test_proceed_allows_local_tools(self):
        gate = self._gate(lambda c: Proceed())
        out = await gate(_input(f"{PREFIX}get_weather"), "t1", None)
        assert _decision(out) == "allow"

    async def test_proceed_leaves_other_tools_to_cli_rules(self):
        gate = self._gate(lambda c: None)
        assert await gate(_input("Bash"), "t1", None) == {}

    async def test_hook_sees_plain_name_and_payload(self):
        seen = []
        gate = self._gate(lambda c: seen.append(c))
        await gate(_input(f"{PREFIX}pay", to="a"), "t1", None)
        ctx = seen[0]
        assert (ctx.step_name, ctx.source, ctx.payload, ctx.tool_call_id) == (
            "pay",
            "local",
            {"to": "a"},
            "t1",
        )

    async def test_async_hooks_are_awaited(self):
        async def deny(ctx):
            return Deny(reason="async no")

        out = await self._gate(deny)(_input(f"{PREFIX}pay"), "t1", None)
        assert out["hookSpecificOutput"]["permissionDecisionReason"] == "async no"

    async def test_first_non_proceed_decision_wins(self):
        gate = self._gate(
            lambda c: Proceed(), lambda c: Deny(reason="first"), lambda c: Deny()
        )
        out = await gate(_input(f"{PREFIX}pay"), "t1", None)
        assert out["hookSpecificOutput"]["permissionDecisionReason"] == "first"

    async def test_require_approval_defers_and_records_details(self):
        gate = self._gate(
            lambda c: RequireApproval(timeout_seconds=5, instructions="i", reason="r")
        )
        out = await gate(_input(f"{PREFIX}pay"), "t1", None)
        assert _decision(out) == "defer"
        assert gate.deferred_approval("t1") == PendingApproval(5, "i", "r")
        assert gate.deferred_approval("t1").to_dict() == {
            "timeout_seconds": 5,
            "instructions": "i",
            "reason": "r",
        }
        assert gate.deferred_approval("other") is None

    async def test_require_approval_without_call_id_denies(self):
        gate = self._gate(lambda c: RequireApproval())
        assert _decision(await gate(_input(f"{PREFIX}pay"), None, None)) == "deny"

    async def test_call_id_falls_back_to_hook_input(self):
        gate = self._gate(lambda c: RequireApproval())
        payload = {**_input(f"{PREFIX}pay"), "tool_use_id": "t7"}
        assert _decision(await gate(payload, None, None)) == "defer"
        assert gate.deferred_approval("t7") is not None

    async def test_deny_blocks(self):
        out = await self._gate(lambda c: Deny())(_input(f"{PREFIX}x"), "1", None)
        assert _decision(out) == "deny"
        assert out["hookSpecificOutput"]["permissionDecisionReason"] == (
            "Blocked by policy"
        )

    async def test_skip_on_bound_tool_returns_its_result(self):
        gate = self._gate(lambda c: Skip(result={"cached": 1}))
        out = await gate(_input(f"{PREFIX}x", a=1), "2", None)
        assert _decision(out) == "allow"
        assert gate.take_skipped("x", {"a": 1}) == '{"cached": 1}'
        assert gate.take_skipped("x", {"a": 1}) is None

    async def test_skip_on_other_tool_denies(self):
        gate = self._gate(lambda c: Skip(result="cached"))
        out = await gate(_input("mcp__github__issue"), "3", None)
        assert _decision(out) == "deny"
        assert "cached" in out["hookSpecificOutput"]["permissionDecisionReason"]

    async def test_mutate_rewrites_input(self):
        gate = self._gate(lambda c: Mutate(payload={"to": "b"}))
        out = await gate(_input(f"{PREFIX}pay", to="a"), "1", None)
        assert _decision(out) == "allow"
        assert out["hookSpecificOutput"]["updatedInput"] == {"to": "b"}

    async def test_failing_hook_fails_closed(self):
        def explode(ctx):
            raise ValueError("bad hook")

        out = await self._gate(explode)(_input(f"{PREFIX}pay"), "1", None)
        assert _decision(out) == "deny"
        assert "bad hook" in out["hookSpecificOutput"]["permissionDecisionReason"]

    async def test_decisions_apply_to_their_exact_call_id_only(self):
        calls = []
        gate = self._gate(
            lambda c: calls.append(c) or RequireApproval(),
            decisions={
                "t1": ToolCallDecision("t1", True),
                "t2": ToolCallDecision("t2", False),
                "t3": ToolCallDecision("t3", False, "nope"),
            },
        )
        assert _decision(await gate(_input(f"{PREFIX}pay"), "t1", None)) == "allow"
        denied = await gate(_input(f"{PREFIX}pay"), "t2", None)
        assert denied["hookSpecificOutput"]["permissionDecisionReason"] == (
            "Rejected by human approver"
        )
        custom = await gate(_input(f"{PREFIX}pay"), "t3", None)
        assert custom["hookSpecificOutput"]["permissionDecisionReason"] == "nope"
        # Decided calls skip the hooks; a re-issued call is evaluated again.
        assert calls == []
        assert _decision(await gate(_input(f"{PREFIX}pay"), "t4", None)) == "defer"


class TestApprovedArguments:
    def _gate(self, digest):
        decision = ToolCallDecision("t1", True, arguments_digest=digest)
        return ToolGate(hooks=(), decisions={"t1": decision}, tool_prefix=PREFIX)

    async def test_matching_arguments_are_allowed(self):
        gate = self._gate(arguments_digest({"to": "a"}))
        assert _decision(await gate(_input(f"{PREFIX}pay", to="a"), "t1", None)) == (
            "allow"
        )

    async def test_changed_arguments_are_denied(self):
        gate = self._gate(arguments_digest({"to": "a"}))
        out = await gate(_input(f"{PREFIX}pay", to="mallory"), "t1", None)
        assert _decision(out) == "deny"
