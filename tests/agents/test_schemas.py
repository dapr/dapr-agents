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

from dapr_agents.agents.schemas import (
    ApprovalRequiredEvent,
    ApprovalResponseEvent,
    CallerClaims,
    TriggerAction,
)


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


def test_approval_response_event_token_excluded_from_repr():
    # approver_token is a sensitive raw JWT and must not leak via repr / logs,
    # but it must still serialize for delivery.
    r = ApprovalResponseEvent(
        approval_request_id="appr-1",
        approved=True,
        approver_token="eyJhbG...",
    )
    assert "eyJhbG..." not in repr(r)
    assert r.model_dump(mode="json")["approver_token"] == "eyJhbG..."


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


# ----- TriggerAction -----


def test_trigger_action_backwards_compat():
    t = TriggerAction(task="say hello")
    assert t.task == "say hello"
    assert t.workflow_instance_id is None
    assert t.caller_headers is None


def test_trigger_action_with_caller_headers():
    headers = {
        "Authorization": "Bearer eyJhbG...",
        "X-Request-Id": "req-c4d2e1f0",
        "traceparent": "00-4bf92f3577b34da6-a3ce929d0e0e4736-01",
    }
    t = TriggerAction(task="x", caller_headers=headers)
    assert t.caller_headers["Authorization"].startswith("Bearer ")
    assert t.caller_headers["X-Request-Id"] == "req-c4d2e1f0"


def test_trigger_action_caller_headers_excluded_from_serialization():
    # caller_headers is transient (exclude=True): it must never be emitted by
    # model_dump(), so raw headers like Authorization cannot leak into pub/sub
    # payloads, signed workflow history, or logs.
    t = TriggerAction(
        task="x",
        workflow_instance_id="wf-1",
        caller_headers={"Authorization": "Bearer x"},
    )
    payload = t.model_dump()
    assert "caller_headers" not in payload
    assert payload["task"] == "x"
    assert payload["workflow_instance_id"] == "wf-1"
    assert payload.get("caller_claims") is None

    # Round-tripping the serialized payload drops the transient headers.
    t2 = TriggerAction.model_validate(payload)
    assert t2.task == "x"
    assert t2.workflow_instance_id == "wf-1"
    assert t2.caller_headers is None


def test_trigger_action_caller_headers_excluded_even_when_omitted():
    t = TriggerAction(task="x")
    payload = t.model_dump()
    assert "caller_headers" not in payload


def test_trigger_action_empty_headers_dict():
    # Empty dict allowed; semantically equivalent to None for plugin behavior
    t = TriggerAction(task="x", caller_headers={})
    assert t.caller_headers == {}


# ----- TriggerAction.caller_claims -----


def test_trigger_action_caller_claims_default_none():
    ta = TriggerAction(task="x")
    assert ta.caller_claims is None


def test_trigger_action_caller_claims_populated():
    ta = TriggerAction(
        task="x",
        caller_claims=CallerClaims(
            subject="alice@acme.com",
            tenant="acme",
            scopes=["agent.invoke"],
            issuer_id="trusted-issuer",
        ),
    )
    assert ta.caller_claims.subject == "alice@acme.com"
    assert "agent.invoke" in ta.caller_claims.scopes


def test_trigger_action_caller_claims_serialization_roundtrip():
    ta = TriggerAction(
        task="x",
        caller_claims=CallerClaims(subject="alice", scopes=["s1"]),
    )
    payload = ta.model_dump()
    assert payload["caller_claims"]["subject"] == "alice"
    assert payload["caller_claims"]["scopes"] == ["s1"]
    ta2 = TriggerAction.model_validate(payload)
    assert ta2.caller_claims.subject == "alice"


def test_caller_claims_in_signed_workflow_history():
    # caller_claims serializes (unlike caller_headers which is excluded).
    # Important for audit lineage + PropagateLineage propagation.
    ta = TriggerAction(
        task="x",
        caller_headers={"Authorization": "Bearer secret"},  # should NOT serialize
        caller_claims=CallerClaims(subject="alice"),  # should serialize
    )
    payload = ta.model_dump()
    assert "caller_headers" not in payload
    assert payload["caller_claims"]["subject"] == "alice"
    payload_json = ta.model_dump_json()
    assert "Bearer" not in payload_json
    assert "secret" not in payload_json
    assert "alice" in payload_json


def test_caller_claims_no_raw_token_field():
    # Sanity check: CallerClaims doesn't have a field that could carry a raw token.
    fields = CallerClaims.model_fields
    forbidden = {"token", "jwt", "authorization", "bearer"}
    for name in fields:
        assert not any(term in name.lower() for term in forbidden), (
            f"CallerClaims must not have a raw-token field: {name}"
        )
