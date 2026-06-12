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

"""Tests for DurableAgent + MCP-workflow tool wiring.

Covers the dapr-agents-specific concerns:
- DurableAgent workflow/activity registration invariants when MCP tools are present
- Tool type guarantees (WorkflowContextInjectedTool) for MCP vs regular tools
- Mixed tool sets (MCP + plain AgentTool)
- Failure paths through run_tool and the converted MCP tool executor

Workflow scheduling, allowed_tools filtering, and per-server caching are
owned by the python-sdk ``DaprMCPClient`` and tested there.
"""

import json
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import ValidationError

from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.schemas import AgentWorkflowEntry
from dapr_agents.tool.base import AgentTool
from dapr_agents.tool.mcp.dapr_workflow_client import mcp_tool_def_to_workflow_tool
from dapr.ext.workflow import MCPToolDef
from dapr.ext.workflow.mcp_schema import create_pydantic_model_from_schema
from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool
from dapr_agents.types import AgentToolExecutorError, ToolError

_MCP_TOOLS = [
    {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "multiply",
        "description": "Multiply two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
]

_WEATHER_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "inputSchema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
]

_MCP_WORKFLOW_PREFIX = "dapr.internal.mcp."


def _tool_def(server_name: str, raw: dict) -> MCPToolDef:
    """Build an :class:`MCPToolDef` matching python-sdk's wire shape."""
    name = raw.get("name", "")
    return MCPToolDef(
        name=name,
        description=raw.get("description", ""),
        input_schema=raw.get("inputSchema") or {},
        server_name=server_name,
        call_tool_workflow=f"{_MCP_WORKFLOW_PREFIX}{server_name}.CallTool.{name}",
    )


def _make_mcp_tools(
    server_name: str = "math-server", tool_defs: list = None
) -> List[WorkflowContextInjectedTool]:
    """Build the WorkflowContextInjectedTool list a dapr-agents caller would
    get after running :class:`DaprMCPClient` + ``mcp_tool_def_to_workflow_tool``."""
    if tool_defs is None:
        tool_defs = _MCP_TOOLS
    return [
        mcp_tool_def_to_workflow_tool(_tool_def(server_name, td)) for td in tool_defs
    ]


def _make_completed_state(output: dict) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_status=SimpleNamespace(name="COMPLETED"),
        serialized_output=json.dumps(output),
    )


def _make_regular_tool() -> AgentTool:
    """A plain AgentTool (not WorkflowContextInjectedTool)."""
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    args_model = create_pydantic_model_from_schema(schema, "EchoArgs")
    return AgentTool(
        name="echo",
        description="Echo the input text.",
        args_model=args_model,
        func=lambda text: text,
    )


def _make_agent(tools: list, name: str = "TestAgent") -> DurableAgent:
    from dapr_agents.agents.configs import AgentPubSubConfig, AgentStateConfig
    from dapr_agents.storage.daprstores.stateservice import StateStoreService

    return DurableAgent(
        name=name,
        role="Test Assistant",
        goal="Run tests",
        instructions=["Follow test instructions"],
        tools=tools,
        pubsub=AgentPubSubConfig(pubsub_name="testpubsub"),
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
    )


# autouse fixture: suppress Dapr state-store I/O for all tests in this file
@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
    monkeypatch.setattr(DurableAgent, "save_state", lambda self: None)

    import dapr_agents.storage.daprstores.statestore as statestore

    class MockDaprClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def save_state(self, *args, **kwargs):
            pass

        def get_state(self, *args, **kwargs):
            class R:
                data = "{}"
                etag = "etag"

            return R()

        def execute_state_transaction(self, *args, **kwargs):
            pass

        def get_metadata(self):
            response = MagicMock()
            response.registered_components = []
            response.application_id = "test-app-id"
            return response

    statestore.DaprClient = MockDaprClient
    monkeypatch.setattr(DurableAgent, "register_agentic_system", lambda self: None)
    yield


class TestWorkflowRegistration:
    """DurableAgent must register consistent workflow + activity sets."""

    def _registered_workflow_names(self, agent: DurableAgent) -> set:
        mock_runtime = MagicMock()
        agent.register_workflows(mock_runtime)
        return {
            call.args[0].__name__
            for call in mock_runtime.register_workflow.call_args_list
            if call.args
        }

    def _registered_activity_names(self, agent: DurableAgent) -> set:
        mock_runtime = MagicMock()
        agent.register_workflows(mock_runtime)
        return {
            # Activities are registered with an agent-scoped prefix
            # (e.g. ``dapr.agents.<name>.run_tool``); strip it for assertions.
            call.args[0].__name__.rsplit(".", 1)[-1]
            for call in mock_runtime.register_activity.call_args_list
            if call.args
        }

    def test_agent_workflow_registered_with_mcp_tools(self):
        agent = _make_agent(_make_mcp_tools())
        wf_names = self._registered_workflow_names(agent)
        assert agent.agent_workflow_name in wf_names

    def test_agent_workflow_registered_with_no_tools(self):
        agent = _make_agent([])
        wf_names = self._registered_workflow_names(agent)
        assert agent.agent_workflow_name in wf_names

    def test_agent_workflow_registered_with_mixed_tools(self):
        agent = _make_agent(_make_mcp_tools() + [_make_regular_tool()])
        wf_names = self._registered_workflow_names(agent)
        assert agent.agent_workflow_name in wf_names

    def test_run_tool_activity_always_registered(self):
        agent = _make_agent(_make_mcp_tools())
        activity_names = self._registered_activity_names(agent)
        assert "run_tool" in activity_names

    def test_call_llm_activity_always_registered(self):
        agent = _make_agent(_make_mcp_tools())
        activity_names = self._registered_activity_names(agent)
        assert "call_llm" in activity_names

    def test_all_core_activities_registered(self):
        agent = _make_agent(_make_mcp_tools())
        activity_names = self._registered_activity_names(agent)
        core = {
            "record_initial_entry",
            "call_llm",
            "run_tool",
            "save_tool_results",
            "finalize_workflow",
            "get_team_members",
            "load_tools",
        }
        missing = core - activity_names
        assert not missing, f"Missing activities: {missing}"

    def test_registration_count_same_with_or_without_mcp_tools(self):
        """MCP tools must not affect how many workflows/activities are registered."""
        mock_a = MagicMock()
        mock_b = MagicMock()

        agent_no_tools = _make_agent([])
        agent_no_tools.register_workflows(mock_a)

        agent_mcp = _make_agent(_make_mcp_tools())
        agent_mcp.register_workflows(mock_b)

        assert (
            mock_a.register_workflow.call_count == mock_b.register_workflow.call_count
        )
        assert (
            mock_a.register_activity.call_count == mock_b.register_activity.call_count
        )


class TestMCPToolConversion:
    """``mcp_tool_def_to_workflow_tool`` must produce LLM-ready tools."""

    def test_tools_are_workflow_context_injected(self):
        for tool in _make_mcp_tools():
            assert isinstance(tool, WorkflowContextInjectedTool), (
                f"Expected WorkflowContextInjectedTool, got {type(tool)} for '{tool.name}'"
            )

    def test_regular_tool_is_not_workflow_context_injected(self):
        assert not isinstance(_make_regular_tool(), WorkflowContextInjectedTool)

    def test_tool_schema_excludes_ctx(self):
        for tool in _make_mcp_tools():
            props = tool.to_function_call()["function"]["parameters"]["properties"]
            assert "ctx" not in props, (
                f"'ctx' must not appear in LLM schema for '{tool.name}'"
            )

    def test_tool_schema_excludes_hidden_kwargs(self):
        for tool in _make_mcp_tools():
            props = tool.to_function_call()["function"]["parameters"]["properties"]
            assert "_child_instance_id" not in props
            assert "_source_agent" not in props

    def test_tool_schema_contains_declared_params(self):
        tools = _make_mcp_tools()
        add = next(t for t in tools if t.name == "add")
        props = add.to_function_call()["function"]["parameters"]["properties"]
        assert "a" in props
        assert "b" in props

    def test_tool_description_preserved(self):
        tools = _make_mcp_tools()
        add = next(t for t in tools if t.name == "add")
        assert add.description == "Add two integers."

    def test_multi_server_tools_combine_correctly(self):
        """Converting tool defs from two servers produces the full tool set."""
        tools = _make_mcp_tools("math-server", _MCP_TOOLS) + _make_mcp_tools(
            "weather-server", _WEATHER_TOOLS
        )
        names = {t.name for t in tools}
        assert names == {"add", "multiply", "get_weather"}


class TestMixedToolSet:
    """Agents must work correctly when tool list contains both tool types."""

    def test_agent_registers_both_tool_types_in_executor(self):
        agent = _make_agent(_make_mcp_tools() + [_make_regular_tool()])

        tool_names = {t.name for t in agent.tool_executor.tools}
        assert "add" in tool_names
        assert "multiply" in tool_names
        assert "echo" in tool_names

    def test_duplicate_tool_name_raises_when_passed_directly(self):
        """Passing two MCP tools with the same name directly to the agent
        constructor raises AgentToolExecutorError — duplicates must be loud.
        The MCP auto-discovery path (``connect_mcpservers``) checks
        ``get_tool()`` to stay idempotent across restarts."""
        tools_a = _make_mcp_tools("server-a", _MCP_TOOLS)  # add, multiply
        tools_b = _make_mcp_tools("server-b", _MCP_TOOLS)  # add, multiply

        with pytest.raises(AgentToolExecutorError, match="already registered"):
            _make_agent(tools_a + tools_b)

    def test_mcp_tools_remain_workflow_context_injected_in_agent(self):
        agent = _make_agent(_make_mcp_tools() + [_make_regular_tool()])

        for tool in agent.tool_executor.tools:
            if tool.name in {"add", "multiply"}:
                assert isinstance(tool, WorkflowContextInjectedTool)
            elif tool.name == "echo":
                assert not isinstance(tool, WorkflowContextInjectedTool)

    def test_run_tool_activity_executes_regular_tool(self):
        """run_tool activity can execute a regular (non-MCP) tool."""
        regular = _make_regular_tool()
        agent = _make_agent([regular])

        entry = AgentWorkflowEntry(
            source=None,
            triggering_workflow_instance_id=None,
            messages=[],
            tool_history=[],
        )
        agent._infra._state_model = entry

        mock_ctx = Mock()
        with (
            patch.object(agent._infra, "load_state"),
            patch.object(agent, "save_state"),
        ):
            result = agent.run_tool(
                mock_ctx,
                {
                    "instance_id": "test-123",
                    "tool_call": {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": '{"text": "hello"}',
                        },
                    },
                },
            )

        assert result["tool_call_id"] == "call_abc"
        assert result["name"] == "echo"


class TestMCPToolExecutor:
    """The converted tool's executor schedules the correct child workflow."""

    def test_call_tool_invalid_args_raises_validation_error(self):
        """Args that don't match the tool schema must raise before child workflow."""
        tools = _make_mcp_tools()
        ctx = MagicMock()

        add = next(t for t in tools if t.name == "add")

        with pytest.raises((ValidationError, ToolError)):
            add(ctx=ctx, a="definitely-not-an-int", b="also-not")

        # The child workflow must NOT have been called since validation failed
        ctx.call_child_workflow.assert_not_called()

    def test_call_tool_child_workflow_exception_propagates(self):
        tools = _make_mcp_tools()
        ctx = MagicMock()
        ctx.call_child_workflow.side_effect = RuntimeError("sidecar unavailable")

        add = next(t for t in tools if t.name == "add")

        with pytest.raises((RuntimeError, ToolError), match="sidecar unavailable"):
            add(ctx=ctx, a=1, b=2)

    def test_call_tool_schedules_per_tool_workflow_name(self):
        """The executor must use the per-tool CallTool workflow name."""
        tools = _make_mcp_tools("math-server", _MCP_TOOLS)
        add = next(t for t in tools if t.name == "add")

        ctx = MagicMock()
        add(ctx=ctx, a=1, b=2)

        ctx.call_child_workflow.assert_called_once()
        call_kwargs = ctx.call_child_workflow.call_args.kwargs
        assert (
            call_kwargs["workflow"] == f"{_MCP_WORKFLOW_PREFIX}math-server.CallTool.add"
        )
        assert call_kwargs["input"] == {"arguments": {"a": 1, "b": 2}}


class TestRunToolFailureScenarios:
    """run_tool activity error handling for non-MCP tools."""

    def test_run_tool_returns_error_tool_message_on_tool_exception(self):
        """run_tool must return a ToolMessage dict even when the tool raises."""

        # Build an agent with a tool whose func raises
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        args_model = create_pydantic_model_from_schema(schema, "BoomArgs")

        def boom(x: int):
            raise ValueError("boom!")

        boom_tool = AgentTool(
            name="boom",
            description="Always fails.",
            args_model=args_model,
            func=boom,
        )
        agent = _make_agent([boom_tool])

        entry = AgentWorkflowEntry(
            source=None,
            triggering_workflow_instance_id=None,
            messages=[],
            tool_history=[],
        )
        agent._infra._state_model = entry

        mock_ctx = Mock()
        with (
            patch.object(agent._infra, "load_state"),
            patch.object(agent, "save_state"),
        ):
            result = agent.run_tool(
                mock_ctx,
                {
                    "instance_id": "test-fail",
                    "tool_call": {
                        "id": "call_fail",
                        "type": "function",
                        "function": {"name": "boom", "arguments": '{"x": 1}'},
                    },
                },
            )

        # Must return a dict (ToolMessage) — never re-raise
        assert isinstance(result, dict)
        assert result["tool_call_id"] == "call_fail"
        assert result["name"] == "boom"
        # Content should contain some indication of the error
        assert result.get("content") is not None

    def test_run_tool_unknown_tool_name_returns_error_tool_message(self):
        """run_tool with an unknown tool name must not crash the orchestrator."""
        agent = _make_agent([])

        entry = AgentWorkflowEntry(
            source=None,
            triggering_workflow_instance_id=None,
            messages=[],
            tool_history=[],
        )
        agent._infra._state_model = entry

        mock_ctx = Mock()
        with (
            patch.object(agent._infra, "load_state"),
            patch.object(agent, "save_state"),
        ):
            result = agent.run_tool(
                mock_ctx,
                {
                    "instance_id": "test-unknown",
                    "tool_call": {
                        "id": "call_unknown",
                        "type": "function",
                        "function": {
                            "name": "nonexistent_tool",
                            "arguments": "{}",
                        },
                    },
                },
            )

        assert isinstance(result, dict)
        assert result["tool_call_id"] == "call_unknown"


# ---------------------------------------------------------------------------
# TestCallToolResultSerialization
# Verifies that serialize_tool_result handles every realistic shape that the
# dapr.internal.mcp.<server>.CallTool.<tool> child workflow might return.
#
# Context: WorkflowContextInjectedTool results bypass run_tool entirely.
# The orchestrator receives the raw child-workflow return value and passes it
# directly to serialize_tool_result() before wrapping it in a ToolMessage.
# This is the only place that transformation is tested.
# ---------------------------------------------------------------------------


class TestCallToolResultSerialization:
    """serialize_tool_result handles every realistic CallTool return shape."""

    def _serialize(self, value):
        from dapr_agents.tool.utils.serialization import serialize_tool_result

        return serialize_tool_result(value)

    # -- Plain string (most common: sidecar returns text content directly) --

    def test_string_result_returned_as_is(self):
        assert self._serialize("sunny, 72°F") == "sunny, 72°F"

    def test_empty_string_result(self):
        assert self._serialize("") == ""

    # -- Dict result (structured JSON from CallTool workflow) --

    def test_dict_result_serializes_to_json(self):
        result = {"content": [{"type": "text", "text": "sunny, 72°F"}]}
        serialized = self._serialize(result)
        assert json.loads(serialized) == result

    def test_nested_dict_result_round_trips(self):
        result = {"result": 42, "unit": "°F", "location": "Seattle"}
        assert json.loads(self._serialize(result)) == result

    # -- None / null --

    def test_none_result_serializes_to_null(self):
        assert self._serialize(None) == "null"

    # -- Numeric primitives --

    def test_integer_result_serializes(self):
        assert self._serialize(5) == "5"

    def test_float_result_serializes(self):
        assert self._serialize(3.14) == "3.14"

    # -- List result (multi-content block shape) --

    def test_list_of_dicts_serializes(self):
        result = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert json.loads(self._serialize(result)) == result

    def test_empty_list_serializes(self):
        assert self._serialize([]) == "[]"

    # -- Ensures ToolMessage.content is always a non-None str --

    def test_serialized_result_is_always_str(self):
        for value in ("text", 42, None, {"k": "v"}, [1, 2], True):
            result = self._serialize(value)
            assert isinstance(result, str), (
                f"Expected str for {value!r}, got {type(result)}"
            )
