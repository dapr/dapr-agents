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

import pytest
from dapr_agents.lifecycle import LifecycleDispatcher, DecisionDict


def test_protocol_accepts_minimal_implementation():
    """A minimal class that implements the 3 methods satisfies the Protocol."""

    class FakeDispatcher:
        def attach(self, agent):
            pass

        def detach(self):
            pass

        def dispatch(self, event_name, context):
            return None

    d = FakeDispatcher()
    assert isinstance(d, LifecycleDispatcher)


def test_protocol_rejects_incomplete_implementation():
    """A class missing one of the 3 methods does NOT satisfy the Protocol
    under runtime_checkable."""

    class IncompleteDispatcher:
        def attach(self, agent):
            pass

        # missing detach and dispatch

    d = IncompleteDispatcher()
    assert not isinstance(d, LifecycleDispatcher)


def test_protocol_dispatch_returns_optional_decision():
    """dispatch() can return None or a DecisionDict."""

    class Dispatcher:
        def attach(self, agent):
            pass

        def detach(self):
            pass

        def dispatch(self, event_name, context):
            if event_name == "deny_me":
                return {"type": "deny", "code": "test.denied"}
            return None

    d = Dispatcher()
    assert d.dispatch("anything", {}) is None
    result = d.dispatch("deny_me", {})
    assert result["type"] == "deny"
    assert result["code"] == "test.denied"


# ----- DecisionDict shape -----


def test_decision_dict_proceed():
    decision: DecisionDict = {"type": "proceed"}
    assert decision["type"] == "proceed"


def test_decision_dict_deny_with_code_details():
    decision: DecisionDict = {
        "type": "deny",
        "code": "oauth.invalid_signature",
        "details": {"issuer": "https://example.com/"},
    }
    assert decision["code"] == "oauth.invalid_signature"


def test_decision_dict_require_approval():
    decision: DecisionDict = {
        "type": "require_approval",
        "required_approver_scopes": ["approver.delete"],
        "approver_audience": "agent-svc",
    }
    assert decision["required_approver_scopes"] == ["approver.delete"]


# ----- No forbidden imports -----


def test_lifecycle_module_has_no_external_deps():
    """The lifecycle module must stay OSS-clean — only stdlib imports."""
    import dapr_agents.lifecycle as lifecycle_mod
    import inspect

    source = inspect.getsource(lifecycle_mod)
    # No proprietary or plugin-specific imports
    for forbidden in ("diagrid", "catalyst", "cloudgrid"):
        assert forbidden not in source.lower(), (
            f"dapr_agents.lifecycle must not import from {forbidden}"
        )
    # No dapr_agents.hooks imports (Protocol stays decoupled)
    assert "from dapr_agents.hooks" not in source
    assert "import dapr_agents.hooks" not in source
