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

"""Unit tests for the hook-based human-in-the-loop (HITL) system."""

import pytest
from datetime import timezone

from dapr_agents.agents.schemas import ApprovalRequiredEvent, ApprovalResponseEvent
from dapr_agents.agents.configs import AgentApprovalConfig, AgentExecutionConfig
from dapr_agents.hooks import (
    Hooks,
    HookContext,
    Proceed,
    Skip,
    Modify,
    RequireApproval,
    Deny,
)
from dapr_agents.tool.base import AgentTool
from dapr_agents.tool import tool


class TestApprovalRequiredEvent:
    def test_step_name_stored(self):
        event = ApprovalRequiredEvent(
            approval_request_id="req-1",
            instance_id="inst-1",
            step_name="DeleteOldData",
            tool_call_id="call-1",
            tool_arguments={"dataset": "sales-2023"},
            timeout_seconds=120,
        )
        assert event.step_name == "DeleteOldData"
        assert event.step_kind == "tool"
        assert event.source == "local"

    def test_tool_name_compat_property(self):
        # existing code that reads event.tool_name still works
        event = ApprovalRequiredEvent(
            approval_request_id="req-1",
            instance_id="inst-1",
            step_name="DeleteOldData",
            tool_call_id="call-1",
            tool_arguments={},
            timeout_seconds=60,
        )
        assert event.tool_name == "DeleteOldData"

    def test_old_tool_name_kwarg_still_accepted(self):
        # existing construction code using tool_name= still works
        event = ApprovalRequiredEvent(
            approval_request_id="req-1",
            instance_id="inst-1",
            tool_name="DeleteOldData",
            tool_call_id="call-1",
            tool_arguments={"dataset": "sales-2023"},
            timeout_seconds=120,
        )
        assert event.step_name == "DeleteOldData"
        assert event.tool_name == "DeleteOldData"

    def test_instructions_field(self):
        event = ApprovalRequiredEvent(
            approval_request_id="req-1",
            instance_id="inst-1",
            step_name="drop_table",
            tool_call_id="c-1",
            tool_arguments={},
            timeout_seconds=60,
            instructions="confirm you want to drop this table",
        )
        assert event.instructions == "confirm you want to drop this table"

    def test_source_and_step_kind_custom_values(self):
        event = ApprovalRequiredEvent(
            approval_request_id="req-1",
            instance_id="inst-1",
            step_name="delete_repo",
            step_kind="tool",
            source="mcp",
            tool_call_id="c-1",
            tool_arguments={},
            timeout_seconds=60,
        )
        assert event.source == "mcp"
        assert event.step_kind == "tool"

    def test_context_defaults_to_empty_dict(self):
        event = ApprovalRequiredEvent(
            approval_request_id="req-1",
            instance_id="inst-1",
            step_name="T",
            tool_call_id="c-1",
            tool_arguments={},
            timeout_seconds=60,
        )
        assert event.context == {}

    def test_requested_at_is_utc(self):
        event = ApprovalRequiredEvent(
            approval_request_id="req-1",
            instance_id="inst-1",
            step_name="T",
            tool_call_id="c-1",
            tool_arguments={},
            timeout_seconds=60,
        )
        assert event.requested_at.tzinfo is not None
        assert event.requested_at.tzinfo == timezone.utc

    def test_roundtrip_serialization(self):
        event = ApprovalRequiredEvent(
            approval_request_id="req-1",
            instance_id="inst-1",
            step_name="DeleteOldData",
            source="mcp",
            tool_call_id="call-1",
            tool_arguments={"dataset": "sales-2023"},
            timeout_seconds=300,
        )
        data = event.model_dump(mode="json")
        restored = ApprovalRequiredEvent(**data)
        assert restored.approval_request_id == event.approval_request_id
        assert restored.step_name == event.step_name
        assert restored.source == event.source
        assert restored.tool_arguments == event.tool_arguments
        assert restored.timeout_seconds == event.timeout_seconds


class TestApprovalResponseEvent:
    def test_approved_true(self):
        resp = ApprovalResponseEvent(
            approval_request_id="req-1",
            approved=True,
        )
        assert resp.approved is True
        assert resp.reason is None

    def test_approved_false_with_reason(self):
        resp = ApprovalResponseEvent(
            approval_request_id="req-1",
            approved=False,
            reason="dataset still in use",
        )
        assert resp.approved is False
        assert resp.reason == "dataset still in use"

    def test_decided_at_is_utc(self):
        resp = ApprovalResponseEvent(
            approval_request_id="req-1",
            approved=True,
        )
        assert resp.decided_at.tzinfo == timezone.utc

    def test_roundtrip_serialization(self):
        resp = ApprovalResponseEvent(
            approval_request_id="req-1",
            approved=True,
            reason="looks good",
        )
        data = resp.model_dump(mode="json")
        restored = ApprovalResponseEvent(**data)
        assert restored.approved == resp.approved
        assert restored.reason == resp.reason


class TestAgentApprovalConfig:
    def test_default_pubsub_is_none(self):
        # pubsub_name defaults to None — HTTP polling is the default delivery mode
        cfg = AgentApprovalConfig()
        assert cfg.pubsub_name is None
        assert cfg.topic == "agent-approval-requests"

    def test_default_timeout(self):
        cfg = AgentApprovalConfig()
        assert cfg.default_timeout_seconds == 300

    def test_custom_values(self):
        cfg = AgentApprovalConfig(
            pubsub_name="my-pubsub",
            topic="my-topic",
            default_timeout_seconds=60,
        )
        assert cfg.pubsub_name == "my-pubsub"
        assert cfg.topic == "my-topic"
        assert cfg.default_timeout_seconds == 60

    def test_pubsub_name_explicit_none_leaves_topic_default(self):
        cfg = AgentApprovalConfig(pubsub_name=None)
        assert cfg.pubsub_name is None
        assert cfg.topic == "agent-approval-requests"


class TestAgentExecutionConfigApprovalField:
    def test_approval_field_present(self):
        cfg = AgentExecutionConfig()
        assert hasattr(cfg, "approval")
        assert isinstance(cfg.approval, AgentApprovalConfig)

    def test_approval_field_accepts_custom_config(self):
        approval = AgentApprovalConfig(default_timeout_seconds=30)
        cfg = AgentExecutionConfig(approval=approval)
        assert cfg.approval.default_timeout_seconds == 30


class TestHookDecisions:
    def test_proceed_is_a_decision(self):
        d = Proceed()
        from dapr_agents.hooks import HookDecision

        assert isinstance(d, HookDecision)

    def test_deny_carries_reason(self):
        d = Deny(reason="schema changes go through dba review")
        assert d.reason == "schema changes go through dba review"

    def test_deny_reason_is_optional(self):
        d = Deny()
        assert d.reason is None

    def test_skip_carries_result(self):
        d = Skip(result={"cached": True})
        assert d.result == {"cached": True}

    def test_skip_result_is_optional(self):
        d = Skip()
        assert d.result is None

    def test_modify_carries_payload(self):
        d = Modify(payload={"new_arg": "value"})
        assert d.payload == {"new_arg": "value"}

    def test_require_approval_defaults(self):
        d = RequireApproval()
        assert d.timeout_seconds is None
        assert d.instructions is None
        assert d.reason is None

    def test_require_approval_with_all_fields(self):
        d = RequireApproval(
            timeout_seconds=3600,
            instructions="confirm deletion",
            reason="destructive operation",
        )
        assert d.timeout_seconds == 3600
        assert d.instructions == "confirm deletion"
        assert d.reason == "destructive operation"


class TestHooksContainer:
    def test_empty_hooks_has_no_callbacks(self):
        h = Hooks()
        assert h.before_tool_call == []
        assert h.after_tool_call == []
        assert h.before_llm_call == []
        assert h.after_llm_call == []

    def test_before_tool_call_registered(self):
        def my_hook(ctx: HookContext):
            return Proceed()

        h = Hooks(before_tool_call=[my_hook])
        assert my_hook in h.before_tool_call

    def test_hook_called_with_context(self):
        received: list = []

        def capturing_hook(ctx: HookContext):
            received.append(ctx)
            return Deny(reason="test")

        h = Hooks(before_tool_call=[capturing_hook])
        ctx = HookContext(
            step_name="drop_table",
            step_kind="tool",
            source="local",
            payload={"table": "users"},
            tool_call_id="call-123",
        )
        decision = h.before_tool_call[0](ctx)
        assert len(received) == 1
        assert received[0].step_name == "drop_table"
        assert isinstance(decision, Deny)

    def test_hook_returning_none_means_proceed(self):
        # None return from a hook is treated as Proceed in the workflow code
        def passthrough(ctx: HookContext):
            return None  # caller coerces this to Proceed

        h = Hooks(before_tool_call=[passthrough])
        result = h.before_tool_call[0](HookContext("tool", "tool", "local", {}, "id"))
        assert result is None  # workflow code handles the None → Proceed coercion


class TestToolDecoratorNoApprovalFields:
    def test_tool_decorator_no_longer_accepts_requires_approval(self):
        # the decorator no longer has requires_approval; just verifying clean creation
        @tool
        def simple_tool(x: int) -> str:
            """A simple tool."""
            return str(x)

        assert not hasattr(simple_tool, "requires_approval")
        assert not hasattr(simple_tool, "approval_timeout_seconds")

    def test_agent_tool_has_source_field(self):
        @tool
        def local_tool(x: str) -> str:
            """A local tool."""
            return x

        assert local_tool.source == "local"

    def test_mcp_tool_has_mcp_source(self):
        # from_mcp sets source="mcp" — test via direct construction
        t = AgentTool(
            name="delete_repo",
            description="delete a repo",
            func=lambda repo: f"deleted {repo}",
            source="mcp",
        )
        assert t.source == "mcp"
