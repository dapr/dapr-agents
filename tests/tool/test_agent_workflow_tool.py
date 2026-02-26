"""Unit tests for AgentWorkflowTool and agent_to_tool."""

from unittest.mock import MagicMock

import pytest

from dapr_agents.tool.workflow.agent_tool import (
    AGENT_WORKFLOW_SUFFIX,
    AgentTaskArgs,
    AgentWorkflowTool,
    _schedule_agent_workflow,
    agent_to_tool,
)
from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool


class TestAgentWorkflowSuffix:
    def test_suffix_constant_value(self):
        assert AGENT_WORKFLOW_SUFFIX == "_agent_workflow"


class TestAgentTaskArgs:
    def test_task_field_required(self):
        with pytest.raises(Exception):
            AgentTaskArgs()  # task is required

    def test_task_field_accepts_string(self):
        args = AgentTaskArgs(task="Bring the lembas bread")
        assert args.task == "Bring the lembas bread"


class TestScheduleAgentWorkflow:
    def test_schedules_with_correct_name(self):
        ctx = MagicMock()
        _schedule_agent_workflow(ctx, task="carry the Ring", agent_name="sam")
        ctx.call_child_workflow.assert_called_once_with(
            workflow="sam_agent_workflow",
            input={"task": "carry the Ring"},
        )

    def test_schedules_cross_app_with_app_id(self):
        ctx = MagicMock()
        _schedule_agent_workflow(
            ctx, task="scout ahead", agent_name="legolas", target_app_id="legolas-app"
        )
        ctx.call_child_workflow.assert_called_once_with(
            workflow="legolas_agent_workflow",
            input={"task": "scout ahead"},
            app_id="legolas-app",
        )

    def test_no_app_id_when_none(self):
        ctx = MagicMock()
        _schedule_agent_workflow(
            ctx, task="help", agent_name="gandalf", target_app_id=None
        )
        call_kwargs = ctx.call_child_workflow.call_args.kwargs
        assert "app_id" not in call_kwargs


class TestAgentToTool:
    def test_returns_agent_workflow_tool(self):
        tool = agent_to_tool("sam", "Logistics expert.")
        assert isinstance(tool, AgentWorkflowTool)

    def test_is_workflow_context_injected(self):
        tool = agent_to_tool("sam", "Logistics expert.")
        assert isinstance(tool, WorkflowContextInjectedTool)

    def test_name_matches_agent_name(self):
        tool = agent_to_tool("frodo", "Ring-bearer.")
        assert tool.name.lower() == "frodo"

    def test_description_stored(self):
        desc = "Sam Gamgee. Goal: Manage provisions."
        tool = agent_to_tool("sam", desc)
        assert tool.description == desc

    def test_target_agent_name_stored(self):
        tool = agent_to_tool("gandalf", "Wizard.")
        assert tool.target_agent_name == "gandalf"

    def test_target_app_id_none_by_default(self):
        tool = agent_to_tool("sam", "Helper.")
        assert tool.target_app_id is None

    def test_target_app_id_stored(self):
        tool = agent_to_tool("sam", "Helper.", target_app_id="sam-app")
        assert tool.target_app_id == "sam-app"

    def test_args_model_is_agent_task_args(self):
        tool = agent_to_tool("sam", "Helper.")
        assert tool.args_model is AgentTaskArgs

    def test_ctx_not_in_function_schema(self):
        """The workflow context (ctx) must never appear in the LLM-visible schema."""
        tool = agent_to_tool("sam", "Helper.")
        schema = tool.to_function_call()
        params = schema["function"]["parameters"]["properties"]
        assert "ctx" not in params
        assert "task" in params

    def test_tool_calls_correct_child_workflow(self):
        """Calling the tool with ctx and task schedules the right child workflow."""
        tool = agent_to_tool("sam", "Helper.")
        ctx = MagicMock()
        tool(ctx=ctx, task="Pack the bags")
        ctx.call_child_workflow.assert_called_once_with(
            workflow="sam_agent_workflow",
            input={"task": "Pack the bags"},
        )

    def test_cross_app_tool_routes_to_app_id(self):
        tool = agent_to_tool("sam", "Helper.", target_app_id="sam-app")
        ctx = MagicMock()
        tool(ctx=ctx, task="Ready the ponies")
        ctx.call_child_workflow.assert_called_once_with(
            workflow="sam_agent_workflow",
            input={"task": "Ready the ponies"},
            app_id="sam-app",
        )
