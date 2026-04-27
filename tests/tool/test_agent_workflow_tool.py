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
        # Valid agent names are preserved verbatim (casing, hyphens, underscores).
        assert agent_workflow_id("sam") == "dapr.agents.sam.workflow"
        assert agent_workflow_id("frodo") == "dapr.agents.frodo.workflow"

    def test_agent_workflow_id_with_full_workflow_name(self):
        """Test that full workflow names are returned as-is."""
        full_name = "dapr.openai.catering-coordinator.workflow"
        assert agent_workflow_id(full_name) == full_name

        full_name2 = "dapr.pydantic_ai.decoration-planner.workflow"
        assert agent_workflow_id(full_name2) == full_name2

        # Legacy format: kebab-case agent names are preserved verbatim
        assert (
            agent_workflow_id("catering-coordinator")
            == "dapr.agents.catering-coordinator.workflow"
        )

    def test_agent_workflow_id_with_framework(self):
        """Test that framework parameter constructs correct workflow names.

        Agent names are preserved verbatim (hyphens/underscores/casing) —
        only characters rejected by the OpenAI/Anthropic tool-name specs
        are stripped.
        """
        assert (
            agent_workflow_id("catering-coordinator", framework="openai")
            == "dapr.openai.catering-coordinator.workflow"
        )

        assert (
            agent_workflow_id("decoration-planner", framework="pydantic_ai")
            == "dapr.pydantic-ai.decoration-planner.workflow"
        )

        assert (
            agent_workflow_id("schedule-planner", framework="langgraph")
            == "dapr.langgraph.schedule-planner.workflow"
        )

        assert (
            agent_workflow_id("venue-scout", framework="crewai")
            == "dapr.crewai.venue-scout.workflow"
        )

        # Dapr Agents framework (should use standard format)
        assert (
            agent_workflow_id("sam", framework="Dapr Agents")
            == "dapr.agents.sam.workflow"
        )

        # None framework (should use standard format)
        assert agent_workflow_id("sam", framework=None) == "dapr.agents.sam.workflow"

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

    def test_agent_workflow_id_with_strands_framework(self):
        """Kebab-case agent names are preserved verbatim."""
        assert (
            agent_workflow_id("strands-default", framework="strands")
            == "dapr.strands.strands-default.workflow"
        )
        assert (
            agent_workflow_id("my-agent", framework="Strands")
            == "dapr.strands.my-agent.workflow"
        )

    def test_agent_workflow_id_langgraph(self):
        assert (
            agent_workflow_id("schedule-planner", framework="CompiledStateGraph")
            == "dapr.compiledstategraph.schedule-planner.workflow"
        )

    @pytest.mark.parametrize(
        "framework",
        [
            "Dapr Agents",
            "dapr agents",
            "DAPR AGENTS",
            "dapr-agents",
            "Dapr-Agents",
            "dapr_agents",
            "Dapr_Agents",
            "dapr.agents",
            "Dapr.Agents",
            "DaprAgents",
            "daprAgents",
        ],
    )
    def test_dapr_agents_framework_aliases_route_to_default(self, framework):
        """All separator/casing variants of 'Dapr Agents' resolve to dapr.agents.*.

        Regression guard for the cross-process dispatch bug: if the orchestrator
        read a registry entry with framework='dapr-agents' (hyphen) or a
        different casing, it used to build 'dapr.dapr-agents.sre-agent.workflow'
        which the sub-agent never registered.
        """
        assert (
            agent_workflow_id("sre-agent", framework=framework)
            == "dapr.agents.sre-agent.workflow"
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

    def test_schedules_with_custom_framework(self):
        ctx = MagicMock()
        _schedule_agent_workflow(
            ctx,
            task="estimate costs",
            agent_name="strands-default",
            framework="strands",
        )
        ctx.call_child_workflow.assert_called_once_with(
            workflow="dapr.strands.strands-default.workflow",
            input={"task": "estimate costs"},
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
        # Hyphens are valid per OpenAI/Anthropic specs and preserved verbatim
        ctx.call_child_workflow.assert_called_once_with(
            workflow="dapr.openai.catering-coordinator.workflow",
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


class TestWorkflowContextInjectedToolError:
    """Error surfaced by WorkflowContextInjectedTool when ctx is missing."""

    def test_error_names_the_tool_and_class(self):
        """Missing ctx error should include tool name and class for diagnosis.

        The old message ('Missing workflow context. Pass it as ...') read like
        a framework bug; it gave the user no way to tell which tool, which
        dispatch path, or why ctx was absent. The enriched message names the
        offending tool so the user can trace it back to the dispatch loop.
        """
        from dapr_agents.types import ToolError

        tool = agent_to_tool("frodo", "Ring-bearer.")
        with pytest.raises(ToolError) as exc_info:
            tool(task="Carry it to Mordor")  # no ctx passed
        msg = str(exc_info.value)
        assert "frodo" in msg.lower() or "Frodo" in msg
        assert "AgentWorkflowTool" in msg
        assert "ctx=<DaprWorkflowContext>" in msg


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
        # Hyphens are valid per OpenAI/Anthropic specs and preserved verbatim
        ctx.call_child_workflow.assert_called_once_with(
            workflow="dapr.openai.catering-coordinator.workflow",
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
        """Test that agent names with spaces have the spaces stripped."""
        tool = agent_to_tool("Randomagee Geegee", "Helper.")
        # Whitespace is stripped; original casing is preserved.
        assert tool.name == "RandomageeGeegee"

        function_call = tool.to_function_call()
        assert function_call["function"]["name"] == "RandomageeGeegee"

    def test_agent_to_tool_sanitizes_name_with_special_chars(self):
        """Test that agent names with invalid characters have them stripped."""
        tool = agent_to_tool("agent<name>", "Test agent.")
        # Only the invalid characters (<, >) are removed; casing preserved.
        assert tool.name == "agentname"

        function_call = tool.to_function_call()
        assert function_call["function"]["name"] == "agentname"

    def test_sanitize_openai_tool_name(self):
        """Test the sanitize_openai_tool_name function directly.

        Provider APIs require tool names to match ``^[a-zA-Z0-9_-]{1,64}$``.
        Any character outside that set is stripped. Hyphens, underscores,
        and the original casing must be preserved so developer-written names
        (e.g. kebab-case YAML keys) reach the LLM unchanged.
        """
        # Characters outside [a-zA-Z0-9_-] are stripped
        assert sanitize_openai_tool_name("Randomagee Geegee") == "RandomageeGeegee"
        assert sanitize_openai_tool_name("agent<name>") == "agentname"
        assert sanitize_openai_tool_name("tool|name") == "toolname"
        assert sanitize_openai_tool_name("tool\\name") == "toolname"
        assert sanitize_openai_tool_name("tool/name") == "toolname"
        assert sanitize_openai_tool_name("tool>name") == "toolname"
        assert sanitize_openai_tool_name("tool  name") == "toolname"

        # Dots, special symbols, non-ASCII are stripped
        assert sanitize_openai_tool_name("my.tool.name") == "mytoolname"
        assert sanitize_openai_tool_name("tool!@#$%^&*()") == "tool"
        assert sanitize_openai_tool_name("café_finder") == "caf_finder"
        assert sanitize_openai_tool_name("search:query") == "searchquery"

        # Hyphens, underscores, and casing are preserved verbatim
        assert sanitize_openai_tool_name("get-xyz-count") == "get-xyz-count"
        assert sanitize_openai_tool_name("get_user") == "get_user"
        assert sanitize_openai_tool_name("valid_name") == "valid_name"
        assert sanitize_openai_tool_name("tool___name") == "tool___name"
        assert sanitize_openai_tool_name("_tool_name_") == "_tool_name_"
        assert sanitize_openai_tool_name("GetUser") == "GetUser"
        assert sanitize_openai_tool_name("RandomageeGeegee") == "RandomageeGeegee"
        assert sanitize_openai_tool_name("UPPERCASE") == "UPPERCASE"

        # Names exceeding 64 characters raise ValueError
        with pytest.raises(ValueError, match="65 characters"):
            sanitize_openai_tool_name("a" * 65)

        # Exactly 64 is fine
        assert sanitize_openai_tool_name("a" * 64) == "a" * 64

        # Fallback cases
        assert sanitize_openai_tool_name("") == "unnamed_tool"
        assert sanitize_openai_tool_name("   ") == "unnamed_tool"
        assert sanitize_openai_tool_name("<|\\/>") == "unnamed_tool"
        assert sanitize_openai_tool_name("...") == "unnamed_tool"
        assert sanitize_openai_tool_name("@#$%") == "unnamed_tool"

    def test_sanitize_openai_tool_name_logs_warning_on_change(self, caplog):
        """A WARNING should be emitted whenever sanitization mutates the name."""
        import logging

        with caplog.at_level(
            logging.WARNING, logger="dapr_agents.tool.utils.function_calling"
        ):
            result = sanitize_openai_tool_name("Randomagee Geegee")
        assert result == "RandomageeGeegee"
        assert any(
            "Randomagee Geegee" in rec.message and "RandomageeGeegee" in rec.message
            for rec in caplog.records
        ), "Expected a WARNING containing both original and sanitized names"

        # No warning when the name is already valid (including kebab-case)
        caplog.clear()
        with caplog.at_level(
            logging.WARNING, logger="dapr_agents.tool.utils.function_calling"
        ):
            assert sanitize_openai_tool_name("get-xyz-count") == "get-xyz-count"
        assert not caplog.records, (
            "Valid kebab-case names must not trigger a sanitization warning"
        )

    def test_tool_with_custom_framework(self):
        tool = agent_to_tool("strands-default", "Budget analyst.", framework="strands")
        ctx = MagicMock()
        tool(ctx=ctx, task="Estimate costs")
        ctx.call_child_workflow.assert_called_once_with(
            workflow="dapr.strands.strands-default.workflow",
            input={"task": "Estimate costs"},
        )
