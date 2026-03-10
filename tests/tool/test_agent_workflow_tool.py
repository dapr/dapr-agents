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

"""Unit tests for AgentWorkflowTool and agent_to_tool."""

from unittest.mock import MagicMock

import pytest

from dapr_agents.tool.workflow.agent_tool import (
    AGENT_WORKFLOW_SUFFIX,
    AgentTaskArgs,
    AgentWorkflowTool,
    _schedule_agent_workflow,
    agent_to_tool,
    agent_workflow_id,
)
from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool
from dapr_agents.tool.utils.function_calling import sanitize_openai_tool_name


class TestAgentWorkflowSuffix:
    def test_suffix_constant_value(self):
        assert AGENT_WORKFLOW_SUFFIX == "_agent_workflow"  # backward compat

    def test_agent_workflow_id(self):
        # Agent names are sanitized to TitleCase for OpenAI compliance
        assert agent_workflow_id("sam") == "dapr.agents.Sam.workflow"
        assert agent_workflow_id("frodo") == "dapr.agents.Frodo.workflow"

    def test_agent_workflow_id_with_full_workflow_name(self):
        """Test that full workflow names are returned as-is."""
        full_name = "dapr.openai.catering-coordinator.workflow"
        assert agent_workflow_id(full_name) == full_name

        full_name2 = "dapr.pydantic_ai.decoration-planner.workflow"
        assert agent_workflow_id(full_name2) == full_name2

        # Legacy format should still work (agent name sanitized to TitleCase)
        assert (
            agent_workflow_id("catering-coordinator")
            == "dapr.agents.CateringCoordinator.workflow"
        )

    def test_agent_workflow_id_with_framework(self):
        """Test that framework parameter constructs correct workflow names."""
        # Agent names are sanitized to TitleCase for OpenAI compliance
        # OpenAI framework
        assert (
            agent_workflow_id("catering-coordinator", framework="openai")
            == "dapr.openai.CateringCoordinator.workflow"
        )

        # Pydantic AI framework
        assert (
            agent_workflow_id("decoration-planner", framework="pydantic_ai")
            == "dapr.pydantic-ai.DecorationPlanner.workflow"
        )

        # LangGraph framework
        assert (
            agent_workflow_id("schedule-planner", framework="langgraph")
            == "dapr.langgraph.SchedulePlanner.workflow"
        )

        # CrewAI framework
        assert (
            agent_workflow_id("venue-scout", framework="crewai")
            == "dapr.crewai.VenueScout.workflow"
        )

        # Dapr Agents framework (should use standard format)
        assert (
            agent_workflow_id("sam", framework="Dapr Agents")
            == "dapr.agents.Sam.workflow"
        )

        # None framework (should use standard format)
        assert agent_workflow_id("sam", framework=None) == "dapr.agents.Sam.workflow"

    def test_agent_workflow_id_with_explicit_workflow_name(self):
        """Test that explicit workflow_name takes precedence."""
        explicit_name = "dapr.custom.framework.agent.workflow"
        assert (
            agent_workflow_id(
                "agent-name", framework="openai", workflow_name=explicit_name
            )
            == explicit_name
        )

        # Even if agent_name is a full workflow name, workflow_name takes precedence
        assert (
            agent_workflow_id(
                "dapr.other.framework.workflow",
                framework="openai",
                workflow_name=explicit_name,
            )
            == explicit_name
        )


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
            workflow=agent_workflow_id("sam"),
            input={"task": "carry the Ring"},
        )

    def test_schedules_cross_app_with_app_id(self):
        ctx = MagicMock()
        _schedule_agent_workflow(
            ctx, task="scout ahead", agent_name="legolas", target_app_id="legolas-app"
        )
        ctx.call_child_workflow.assert_called_once_with(
            workflow=agent_workflow_id("legolas"),
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

    def test_schedules_with_framework(self):
        ctx = MagicMock()
        _schedule_agent_workflow(
            ctx,
            task="coordinate catering",
            agent_name="catering-coordinator",
            framework="openai",
        )
        # Agent name is sanitized to TitleCase for OpenAI compliance
        ctx.call_child_workflow.assert_called_once_with(
            workflow="dapr.openai.CateringCoordinator.workflow",
            input={"task": "coordinate catering"},
        )

    def test_schedules_with_explicit_workflow_name(self):
        ctx = MagicMock()
        _schedule_agent_workflow(
            ctx,
            task="custom task",
            agent_name="agent-name",
            workflow_name="dapr.custom.workflow.name",
        )
        ctx.call_child_workflow.assert_called_once_with(
            workflow="dapr.custom.workflow.name",
            input={"task": "custom task"},
        )


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
            workflow=agent_workflow_id("sam"),
            input={"task": "Pack the bags"},
        )

    def test_cross_app_tool_routes_to_app_id(self):
        tool = agent_to_tool("sam", "Helper.", target_app_id="sam-app")
        ctx = MagicMock()
        tool(ctx=ctx, task="Ready the ponies")
        ctx.call_child_workflow.assert_called_once_with(
            workflow=agent_workflow_id("sam"),
            input={"task": "Ready the ponies"},
            app_id="sam-app",
        )

    def test_agent_to_tool_with_full_workflow_name(self):
        """Test that agent_to_tool works with full workflow names."""
        full_name = "dapr.openai.catering-coordinator.workflow"
        tool = agent_to_tool(full_name, "Catering coordinator.")
        ctx = MagicMock()
        tool(ctx=ctx, task="Plan the menu")
        ctx.call_child_workflow.assert_called_once_with(
            workflow=full_name,
            input={"task": "Plan the menu"},
        )
        assert tool.target_agent_name == full_name

    def test_agent_to_tool_with_framework(self):
        """Test that agent_to_tool constructs workflow names from framework."""
        tool = agent_to_tool(
            "catering-coordinator",
            "Catering coordinator.",
            framework="openai",
        )
        ctx = MagicMock()
        tool(ctx=ctx, task="Plan the menu")
        # Agent name is sanitized to TitleCase for OpenAI compliance
        ctx.call_child_workflow.assert_called_once_with(
            workflow="dapr.openai.CateringCoordinator.workflow",
            input={"task": "Plan the menu"},
        )
        assert tool.target_agent_name == "catering-coordinator"

    def test_agent_to_tool_with_explicit_workflow_name(self):
        """Test that explicit workflow_name takes precedence over framework."""
        tool = agent_to_tool(
            "catering-coordinator",
            "Catering coordinator.",
            framework="openai",
            workflow_name="dapr.custom.framework.workflow",
        )
        ctx = MagicMock()
        tool(ctx=ctx, task="Plan the menu")
        ctx.call_child_workflow.assert_called_once_with(
            workflow="dapr.custom.framework.workflow",
            input={"task": "Plan the menu"},
        )

    def test_agent_to_tool_sanitizes_name_with_spaces(self):
        """Test that agent names with spaces are sanitized for OpenAI compatibility."""
        tool = agent_to_tool("Samwise Gamgee", "Helper.")
        # Tool name should be sanitized to TitleCase (spaces removed, no separators)
        assert tool.name == "SamwiseGamgee"

        # Verify the sanitized name is used in OpenAI function call format
        function_call = tool.to_function_call()
        assert function_call["function"]["name"] == "SamwiseGamgee"

    def test_agent_to_tool_sanitizes_name_with_special_chars(self):
        """Test that agent names with special characters are sanitized."""
        tool = agent_to_tool("agent<name>", "Test agent.")
        # Special characters are removed, name converted to TitleCase
        assert tool.name == "Agentname"

        function_call = tool.to_function_call()
        assert function_call["function"]["name"] == "Agentname"

    def test_sanitize_openai_tool_name(self):
        """Test the sanitize_openai_tool_name function directly."""
        # Names are normalized to TitleCase (no separators) and invalid chars removed
        assert sanitize_openai_tool_name("Samwise Gamgee") == "SamwiseGamgee"
        assert sanitize_openai_tool_name("agent<name>") == "Agentname"
        assert sanitize_openai_tool_name("tool|name") == "Toolname"
        assert sanitize_openai_tool_name("tool\\name") == "Toolname"
        assert sanitize_openai_tool_name("tool/name") == "Toolname"
        assert sanitize_openai_tool_name("tool>name") == "Toolname"
        assert sanitize_openai_tool_name("tool  name") == "ToolName"  # Multiple spaces
        assert (
            sanitize_openai_tool_name("tool___name") == "ToolName"
        )  # Multiple underscores
        assert (
            sanitize_openai_tool_name("_tool_name_") == "ToolName"
        )  # Leading/trailing
        assert sanitize_openai_tool_name("") == "unnamed_tool"  # Empty string
        assert (
            sanitize_openai_tool_name("valid_name") == "ValidName"
        )  # Converted to TitleCase
        assert (
            sanitize_openai_tool_name("get_user") == "GetUser"
        )  # snake_case -> TitleCase
