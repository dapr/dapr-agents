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

from dapr_agents.agents.schemas import ApprovalRequiredEvent, ApprovalResponseEvent


# ----- ApprovalRequiredEvent -----


def test_approval_required_event_backwards_compat():
    e = ApprovalRequiredEvent(
        approval_request_id="appr-1",
        instance_id="wf-1",
        step_name="my_tool",
        tool_call_id="tc-1",
        tool_arguments={"x": 1},
        timeout_seconds=600,
    )
    assert e.required_approver_scopes == []
    assert e.allowed_approver_subjects == []
    assert e.approver_audience is None


def test_approval_required_event_with_authz():
    e = ApprovalRequiredEvent(
        approval_request_id="appr-1",
        instance_id="wf-1",
        step_name="my_tool",
        tool_call_id="tc-1",
        tool_arguments={},
        timeout_seconds=600,
        required_approver_scopes=["approver.delete-customer"],
        allowed_approver_subjects=["bob@acme.com"],
        approver_audience="agent-svc",
    )
    assert "approver.delete-customer" in e.required_approver_scopes
    assert e.allowed_approver_subjects == ["bob@acme.com"]
    assert e.approver_audience == "agent-svc"


def test_approval_required_event_serialization_roundtrip():
    e = ApprovalRequiredEvent(
        approval_request_id="appr-1",
        instance_id="wf-1",
        step_name="my_tool",
        tool_call_id="tc-1",
        tool_arguments={"x": 1},
        timeout_seconds=600,
        required_approver_scopes=["approver.foo"],
    )
    payload = e.model_dump(mode="json")
    e2 = ApprovalRequiredEvent.model_validate(payload)
    assert e2.required_approver_scopes == ["approver.foo"]


def test_approval_required_event_backward_compat_tool_name():
    # The existing model_validator should still convert "tool_name" to "step_name".
    e = ApprovalRequiredEvent(
        approval_request_id="appr-1",
        instance_id="wf-1",
        tool_name="legacy_name",
        tool_call_id="tc-1",
        tool_arguments={},
        timeout_seconds=600,
    )
    assert e.step_name == "legacy_name"
    assert e.tool_name == "legacy_name"


# ----- ApprovalResponseEvent -----


def test_approval_response_event_backwards_compat():
    r = ApprovalResponseEvent(approval_request_id="appr-1", approved=True)
    assert r.approved is True
    assert r.approver_token is None
    assert r.approver_subject is None


def test_approval_response_event_with_approver_fields():
    r = ApprovalResponseEvent(
        approval_request_id="appr-1",
        approved=True,
        approver_token="eyJhbG...",
        approver_subject="bob@acme.com",
    )
    assert r.approver_token == "eyJhbG..."
    assert r.approver_subject == "bob@acme.com"


def test_approval_response_event_serialization_roundtrip():
    r = ApprovalResponseEvent(
        approval_request_id="appr-1",
        approved=True,
        approver_token="eyJhbG...",
        approver_subject="bob@acme.com",
    )
    payload = r.model_dump(mode="json")
    r2 = ApprovalResponseEvent.model_validate(payload)
    assert r2.approver_token == "eyJhbG..."
    assert r2.approver_subject == "bob@acme.com"
