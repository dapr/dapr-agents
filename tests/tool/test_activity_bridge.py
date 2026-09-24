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

"""Tests for calling workflow-backed tools from inside an activity."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from dapr.ext.workflow import MCPToolDef

from dapr_agents.tool import tool
from dapr_agents.tool.mcp.dapr_workflow_client import mcp_tool_def_to_workflow_tool
from dapr_agents.tool.workflow import agent_to_tool
from dapr_agents.tool.workflow.activity_bridge import (
    ActivityToolContext,
    bridge_workflow_tools,
)
from dapr_agents.tool.workflow.ask_user_tool import build_ask_user_tool
from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool
from dapr_agents.types import ToolError


@tool
def get_weather(city: str) -> str:
    """Get weather."""
    return f"sunny in {city}"


def _state(status: str, output: Any = None, failure: Optional[str] = None):
    return SimpleNamespace(
        runtime_status=SimpleNamespace(name=status),
        serialized_output=json.dumps(output) if output is not None else None,
        failure_details=SimpleNamespace(message=failure) if failure else None,
    )


class FakeWorkflowClient:
    """Records scheduled workflows and completes them immediately."""

    def __init__(self, final_status: str = "COMPLETED") -> None:
        self.states: Dict[str, Any] = {}
        self.scheduled: List[tuple] = []
        self.final_status = final_status
        self.schedule_error: Optional[Exception] = None
        self.wait_error: Optional[Exception] = None

    def get_workflow_state(self, instance_id, fetch_payloads=True):
        return self.states.get(instance_id)

    def schedule_new_workflow(self, workflow, input=None, instance_id=None):
        if self.schedule_error is not None:
            raise self.schedule_error
        self.scheduled.append((workflow, input, instance_id))
        output = {"role": "assistant", "content": f"handled {json.dumps(input)}"}
        self.states[instance_id] = _state(self.final_status, output, "child broke")

    def wait_for_workflow_completion(
        self, instance_id, fetch_payloads=True, timeout_in_seconds=0
    ):
        if self.wait_error is not None:
            raise self.wait_error
        return self.states.get(instance_id)


def _context(client, **kwargs):
    defaults = dict(
        client_factory=lambda: client,
        instance_id="parent",
        round=0,
        source_agent="caller",
        stream_context={"root_instance_id": "parent", "depth": 0},
    )
    defaults.update(kwargs)
    return ActivityToolContext(**defaults)


def _mcp_tool():
    return mcp_tool_def_to_workflow_tool(
        MCPToolDef(
            name="add",
            description="Add numbers.",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            server_name="math",
            call_tool_workflow="dapr.internal.mcp.math.CallTool.add",
        )
    )


class TestBridgeSelection:
    def test_plain_tools_pass_through(self):
        tools = bridge_workflow_tools([get_weather], _context(FakeWorkflowClient()))
        assert tools == [get_weather]

    def test_workflow_tools_are_wrapped(self):
        tools = bridge_workflow_tools(
            [agent_to_tool("helper", "Helps"), _mcp_tool()],
            _context(FakeWorkflowClient()),
        )
        assert [t.name for t in tools] == ["helper", "add"]
        assert not any(isinstance(t, WorkflowContextInjectedTool) for t in tools)
        assert "a" in tools[1].args_model.model_json_schema()["properties"]

    def test_ask_user_and_cross_app_agents_are_left_out(self):
        tools = bridge_workflow_tools(
            [
                build_ask_user_tool(lambda *a, **k: None),
                agent_to_tool("far", "Far away", target_app_id="other-app"),
                get_weather,
            ],
            _context(FakeWorkflowClient()),
        )
        assert tools == [get_weather]


class TestBridgedCalls:
    async def test_agent_tool_schedules_child_agent_workflow(self):
        client = FakeWorkflowClient()
        (helper,) = bridge_workflow_tools(
            [agent_to_tool("helper", "Helps")], _context(client)
        )
        out = await helper.arun(task="summarize")
        workflow, payload, _ = client.scheduled[0]
        assert workflow.endswith("helper.workflow")
        assert payload["task"] == "summarize"
        metadata = payload["_message_metadata"]
        assert metadata["source"] == "caller"
        stream = metadata["_stream_context"]
        assert stream["depth"] == 1
        assert stream["parent_agent"] == "caller"
        assert stream["parent_instance_id"] == "parent"
        assert "handled" in out

    async def test_no_stream_context_without_parent_stream(self):
        client = FakeWorkflowClient()
        (helper,) = bridge_workflow_tools(
            [agent_to_tool("helper", "Helps")], _context(client, stream_context=None)
        )
        await helper.arun(task="x")
        payload = client.scheduled[0][1]
        assert "_stream_context" not in payload.get("_message_metadata", {})

    async def test_mcp_tool_schedules_call_tool_workflow(self):
        client = FakeWorkflowClient()
        (add,) = bridge_workflow_tools([_mcp_tool()], _context(client))
        await add.arun(a=1, b=2)
        workflow, payload, _ = client.scheduled[0]
        assert workflow == "dapr.internal.mcp.math.CallTool.add"
        assert payload == {"arguments": {"a": 1, "b": 2}}

    async def test_retried_attempt_reattaches_to_existing_child(self):
        client = FakeWorkflowClient()
        context = _context(client)
        (first,) = bridge_workflow_tools([_mcp_tool()], context)
        await first.arun(a=1, b=2)
        # A new bridge (a retried activity attempt) derives the same id.
        (retry,) = bridge_workflow_tools([_mcp_tool()], context)
        await retry.arun(a=1, b=2)
        assert len(client.scheduled) == 1

    async def test_repeated_identical_call_gets_new_child(self):
        client = FakeWorkflowClient()
        (add,) = bridge_workflow_tools([_mcp_tool()], _context(client))
        await add.arun(a=1, b=2)
        await add.arun(a=1, b=2)
        await add.arun(a=2, b=2)
        ids = [instance_id for _, _, instance_id in client.scheduled]
        assert len(set(ids)) == 3

    async def test_round_changes_child_ids(self):
        client = FakeWorkflowClient()
        (r0,) = bridge_workflow_tools([_mcp_tool()], _context(client, round=0))
        (r1,) = bridge_workflow_tools([_mcp_tool()], _context(client, round=1))
        await r0.arun(a=1, b=2)
        await r1.arun(a=1, b=2)
        assert len(client.scheduled) == 2

    async def test_failed_child_raises_tool_error(self):
        client = FakeWorkflowClient(final_status="FAILED")
        (add,) = bridge_workflow_tools([_mcp_tool()], _context(client))
        with pytest.raises(ToolError, match="child broke"):
            await add.arun(a=1, b=2)

    async def test_timeout_raises_tool_error(self):
        client = FakeWorkflowClient()
        client.wait_error = TimeoutError()
        (add,) = bridge_workflow_tools(
            [_mcp_tool()], _context(client, timeout_seconds=3)
        )
        with pytest.raises(ToolError, match="within 3s"):
            await add.arun(a=1, b=2)

    async def test_missing_child_raises_tool_error(self):
        client = FakeWorkflowClient()
        client.wait_for_workflow_completion = lambda *a, **k: None
        (add,) = bridge_workflow_tools([_mcp_tool()], _context(client))
        with pytest.raises(ToolError, match="not found"):
            await add.arun(a=1, b=2)

    async def test_schedule_race_is_tolerated(self):
        client = FakeWorkflowClient()
        client.schedule_error = RuntimeError("already exists")
        calls = {"n": 0}
        original = client.get_workflow_state

        def get_state(instance_id, fetch_payloads=True):
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            # A concurrent attempt scheduled (and finished) it meanwhile.
            client.states[instance_id] = _state("COMPLETED", {"content": "ok"})
            return original(instance_id)

        client.get_workflow_state = get_state
        (add,) = bridge_workflow_tools([_mcp_tool()], _context(client))
        assert "ok" in await add.arun(a=1, b=2)

    async def test_schedule_failure_is_raised(self):
        client = FakeWorkflowClient()
        client.schedule_error = RuntimeError("sidecar down")
        (add,) = bridge_workflow_tools([_mcp_tool()], _context(client))
        with pytest.raises(Exception, match="sidecar down"):
            await add.arun(a=1, b=2)

    async def test_tool_that_does_not_schedule_a_child_is_rejected(self):
        def not_a_child(ctx, **kwargs):
            return "plain value"

        odd = WorkflowContextInjectedTool(
            name="odd", description="Odd.", func=not_a_child
        )
        (bridged,) = bridge_workflow_tools([odd], _context(FakeWorkflowClient()))
        with pytest.raises(ToolError, match="cannot run inside an activity"):
            await bridged.arun()
