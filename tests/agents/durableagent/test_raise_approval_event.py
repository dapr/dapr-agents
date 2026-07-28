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

"""Tests for the optional approver_token kwarg on DurableAgent.raise_approval_event.

The kwarg is a pure pass-through: it flows into the published ApprovalResponseEvent
so the downstream HITL plugin can verify the approver's JWT. dapr-agents does not
validate, persist, or log the raw token.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import dapr.ext.workflow as wf

from dapr_agents.agents.durable import DurableAgent


def _agent_waiting_for_approval():
    """Minimal stand-in exposing only what raise_approval_event touches on ``self``.

    The workflow is reported RUNNING so the method gets past its terminal-state guard.
    """
    wf_client = MagicMock()
    wf_client.get_workflow_state.return_value = SimpleNamespace(
        runtime_status=wf.WorkflowStatus.RUNNING
    )
    return SimpleNamespace(
        _wf_client=wf_client,
        _pending_approvals={"appr-1": {"instance_id": "wf-1"}},
        _persist_pending_approvals=MagicMock(),
    )


def _raised_event_data(agent):
    """Return the event payload passed to raise_workflow_event."""
    return agent._wf_client.raise_workflow_event.call_args.kwargs["data"]


def test_raise_approval_event_backwards_compat():
    """Existing callers without approver_token still work; token defaults to None."""
    agent = _agent_waiting_for_approval()

    DurableAgent.raise_approval_event(
        agent,
        instance_id="wf-1",
        approval_request_id="appr-1",
        approved=True,
    )

    data = _raised_event_data(agent)
    assert data["approved"] is True
    assert data.get("approver_token") is None


def test_raise_approval_event_with_approver_token():
    """approver_token kwarg flows into the published ApprovalResponseEvent."""
    agent = _agent_waiting_for_approval()

    DurableAgent.raise_approval_event(
        agent,
        instance_id="wf-1",
        approval_request_id="appr-1",
        approved=True,
        approver_token="eyJhbGciOi...",
    )

    data = _raised_event_data(agent)
    assert data["approver_token"] == "eyJhbGciOi..."


def test_raise_approval_event_with_reason_and_token():
    """Both reason and approver_token populate correctly on a denial."""
    agent = _agent_waiting_for_approval()

    DurableAgent.raise_approval_event(
        agent,
        instance_id="wf-1",
        approval_request_id="appr-1",
        approved=False,
        reason="not authorized in this environment",
        approver_token="eyJhbGciOi...",
    )

    data = _raised_event_data(agent)
    assert data["approved"] is False
    assert data["reason"].startswith("not authorized")
    assert data["approver_token"] == "eyJhbGciOi..."


def test_raise_approval_event_does_not_log_token(caplog):
    """The raw token must never appear in log output, even on success."""
    agent = _agent_waiting_for_approval()

    with caplog.at_level("DEBUG"):
        DurableAgent.raise_approval_event(
            agent,
            instance_id="wf-1",
            approval_request_id="appr-1",
            approved=True,
            approver_token="super-secret-jwt",
        )

    assert "super-secret-jwt" not in caplog.text
