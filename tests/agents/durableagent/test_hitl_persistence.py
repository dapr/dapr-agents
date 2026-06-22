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

"""Regression tests for HITL pending-approval persistence.

``_persist_pending_approvals`` / ``_restore_pending_approvals`` must use the
``StateStoreService`` public API (``save`` / ``load``). They previously called
``save_state`` / ``try_get_state``, which ``StateStoreService`` does not
implement — so with a ``before_tool_call`` ``RequireApproval`` hook plus a Dapr
state store, every persist/restore raised ``AttributeError`` (caught and
logged), silently dropping approvals across restarts.

These tests pin the call sites to the correct API. The persist test fails on the
old code: a ``spec``'d mock raises ``AttributeError`` on ``save_state``, the
method swallows it, and ``save`` is therefore never called.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import dapr.ext.workflow as wf

from dapr_agents.agents.durable import DurableAgent
from dapr_agents.hooks import Hooks
from dapr_agents.storage.daprstores.stateservice import StateStoreService

_KEY = "release-agent:pending_approvals"


def _fake_agent(state_store, pending):
    """Minimal stand-in exposing only what persist/restore touch on ``self``."""
    return SimpleNamespace(
        name="release-agent",
        _hooks=Hooks(before_tool_call=[lambda ctx: None]),
        state_store=state_store,
        _pending_approvals=pending,
        _pending_approvals_key=lambda: _KEY,
        _wf_client=MagicMock(),
    )


class TestPendingApprovalPersistenceAPI:
    def test_persist_uses_statestore_save(self):
        store = MagicMock(spec=StateStoreService)
        pending = {"req-1": {"instance_id": "inst-1", "step_name": "publish"}}
        agent = _fake_agent(store, pending)

        DurableAgent._persist_pending_approvals(agent)

        # Must use the real StateStoreService API, not the nonexistent save_state.
        store.save.assert_called_once()
        _, kwargs = store.save.call_args
        assert kwargs["key"] == _KEY
        # The fix passes the dict directly; StateStoreService.save JSON-encodes it.
        assert kwargs["value"] == pending

    def test_restore_uses_statestore_load(self):
        store = MagicMock(spec=StateStoreService)
        store.load.return_value = {}
        agent = _fake_agent(store, {})

        DurableAgent._restore_pending_approvals(agent)

        store.load.assert_called_once()
        _, kwargs = store.load.call_args
        assert kwargs["key"] == _KEY

    def test_restore_repopulates_active_pending_approvals(self):
        store = MagicMock(spec=StateStoreService)
        store.load.return_value = {
            "req-1": {"instance_id": "inst-1", "step_name": "publish"}
        }
        agent = _fake_agent(store, {})
        agent._wf_client.get_workflow_state.return_value = SimpleNamespace(
            runtime_status=wf.WorkflowStatus.RUNNING
        )

        DurableAgent._restore_pending_approvals(agent)

        assert "req-1" in agent._pending_approvals

    def test_persist_is_noop_without_state_store(self):
        agent = _fake_agent(state_store=None, pending={"req-1": {}})
        # Should simply return without raising.
        DurableAgent._persist_pending_approvals(agent)

    def test_statestore_service_lacks_legacy_api(self):
        # Pin the contract that motivated the fix: these methods do not exist.
        assert not hasattr(StateStoreService, "save_state")
        assert not hasattr(StateStoreService, "try_get_state")
        assert hasattr(StateStoreService, "save")
        assert hasattr(StateStoreService, "load")
