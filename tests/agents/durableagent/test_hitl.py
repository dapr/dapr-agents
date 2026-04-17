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

"""Unit tests for human-in-the-loop (HITL) support in DurableAgent."""

import pytest
from datetime import timezone

from dapr_agents.agents.schemas import ApprovalRequiredEvent, ApprovalResponseEvent
from dapr_agents.agents.configs import AgentApprovalConfig, AgentExecutionConfig
from dapr_agents.tool.base import AgentTool
from dapr_agents.tool import tool


class TestApprovalRequiredEvent:
    def test_required_fields_stored(self):
        event = ApprovalRequiredEvent(
            approval_request_id="req-1",
            instance_id="inst-1",
            tool_name="DeleteOldData",
            tool_call_id="call-1",
            tool_arguments={"dataset": "sales-2023"},
            timeout_seconds=120,
        )
        assert event.approval_request_id == "req-1"
        assert event.instance_id == "inst-1"
        assert event.tool_name == "DeleteOldData"
        assert event.tool_call_id == "call-1"
        assert event.tool_arguments == {"dataset": "sales-2023"}
        assert event.timeout_seconds == 120

    def test_context_defaults_to_empty_dict(self):
        event = ApprovalRequiredEvent(
            approval_request_id="req-1",
            instance_id="inst-1",
            tool_name="T",
            tool_call_id="c-1",
            tool_arguments={},
            timeout_seconds=60,
        )
        assert event.context == {}

    def test_requested_at_is_utc(self):
        event = ApprovalRequiredEvent(
            approval_request_id="req-1",
            instance_id="inst-1",
            tool_name="T",
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
            tool_name="DeleteOldData",
            tool_call_id="call-1",
            tool_arguments={"dataset": "sales-2023"},
            timeout_seconds=300,
        )
        data = event.model_dump(mode="json")
        restored = ApprovalRequiredEvent(**data)
        assert restored.approval_request_id == event.approval_request_id
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
    def test_disabled_by_default(self):
        cfg = AgentApprovalConfig()
        assert cfg.enabled is False

    def test_default_pubsub_and_topic(self):
        cfg = AgentApprovalConfig()
        assert cfg.pubsub_name == "dapr-agents-pubsub"
        assert cfg.topic == "agent-approval-requests"

    def test_default_timeout(self):
        cfg = AgentApprovalConfig()
        assert cfg.default_timeout_seconds == 300

    def test_custom_values(self):
        cfg = AgentApprovalConfig(
            enabled=True,
            pubsub_name="my-pubsub",
            topic="my-topic",
            default_timeout_seconds=60,
        )
        assert cfg.enabled is True
        assert cfg.pubsub_name == "my-pubsub"
        assert cfg.topic == "my-topic"
        assert cfg.default_timeout_seconds == 60


class TestAgentExecutionConfigApprovalField:
    def test_approval_field_present_and_disabled_by_default(self):
        cfg = AgentExecutionConfig()
        assert hasattr(cfg, "approval")
        assert isinstance(cfg.approval, AgentApprovalConfig)
        assert cfg.approval.enabled is False

    def test_approval_field_accepts_custom_config(self):
        approval = AgentApprovalConfig(enabled=True, default_timeout_seconds=30)
        cfg = AgentExecutionConfig(approval=approval)
        assert cfg.approval.enabled is True
        assert cfg.approval.default_timeout_seconds == 30


class TestToolDecoratorApprovalMetadata:
    def test_requires_approval_defaults_to_false(self):
        @tool
        def simple_tool(x: int) -> str:
            """A simple tool."""
            return str(x)

        assert simple_tool.requires_approval is False
        assert simple_tool.approval_timeout_seconds is None

    def test_requires_approval_true_stored(self):
        @tool(requires_approval=True)
        def guarded_tool(dataset: str) -> str:
            """Delete a dataset."""
            return f"Deleted {dataset}"

        assert guarded_tool.requires_approval is True
        assert guarded_tool.approval_timeout_seconds is None

    def test_per_tool_timeout_stored(self):
        @tool(requires_approval=True, approval_timeout_seconds=120)
        def guarded_tool_with_timeout(dataset: str) -> str:
            """Delete a dataset with custom timeout."""
            return f"Deleted {dataset}"

        assert guarded_tool_with_timeout.requires_approval is True
        assert guarded_tool_with_timeout.approval_timeout_seconds == 120

    def test_approval_timeout_without_requires_approval(self):
        # approval_timeout_seconds can be set independently; requires_approval stays False
        @tool(approval_timeout_seconds=60)
        def tool_with_timeout_only(x: str) -> str:
            """A tool."""
            return x

        assert tool_with_timeout_only.requires_approval is False
        assert tool_with_timeout_only.approval_timeout_seconds == 60

    def test_agent_tool_direct_construction(self):
        def my_func(x: str) -> str:
            """My tool."""
            return x

        t = AgentTool(
            name="MyTool",
            description="My tool.",
            func=my_func,
            requires_approval=True,
            approval_timeout_seconds=300,
        )
        assert t.requires_approval is True
        assert t.approval_timeout_seconds == 300
