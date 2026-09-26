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

"""Tests for the paused-run extension of the executor contract."""

import pytest

from dapr_agents.agents.executors import (
    CONTEXT_TOOL_DECISIONS,
    AgentExecutorBase,
    EchoAgentExecutor,
    ExecutorBinding,
    ToolCallDecision,
    arguments_digest,
    tool_decisions_from_context,
)
from dapr_agents.agents.executors.event import (
    EVENT_COMPLETE,
    EVENT_ERROR,
    EVENT_PAUSED,
    TERMINAL_EVENT_TYPES,
)


class TestToolCallDecision:
    def test_round_trips_through_dict(self):
        decision = ToolCallDecision("t1", False, "too much money")
        restored = ToolCallDecision.from_dict("t1", decision.to_dict())
        assert restored == decision

    def test_to_dict_is_json_safe_and_omits_id(self):
        assert ToolCallDecision("t1", True).to_dict() == {
            "approved": True,
            "reason": None,
            "arguments_digest": None,
        }

    def test_arguments_digest_round_trips_and_ignores_key_order(self):
        digest = arguments_digest({"to": "a", "amount": 5})
        assert digest == arguments_digest({"amount": 5, "to": "a"})
        decision = ToolCallDecision("t1", True, arguments_digest=digest)
        assert ToolCallDecision.from_dict("t1", decision.to_dict()) == decision

    @pytest.mark.parametrize("value", ["yes", 1, None])
    def test_only_literal_true_approves(self, value):
        # Fail closed: anything but ``True`` is a rejection.
        assert ToolCallDecision.from_dict("t", {"approved": value}).approved is False

    def test_reason_is_stringified(self):
        assert ToolCallDecision.from_dict("t", {"reason": 42}).reason == "42"

    def test_is_frozen(self):
        decision = ToolCallDecision("t1", True)
        with pytest.raises(AttributeError):
            decision.approved = False  # type: ignore[misc]


class TestToolDecisionsFromContext:
    def test_extracts_decisions_keyed_by_call_id(self):
        context = {
            CONTEXT_TOOL_DECISIONS: {
                "t1": {"approved": True},
                "t2": {"approved": False, "reason": "no"},
            },
            "other": 1,
        }
        decisions = tool_decisions_from_context(context)
        assert decisions == {
            "t1": ToolCallDecision("t1", True),
            "t2": ToolCallDecision("t2", False, "no"),
        }

    @pytest.mark.parametrize(
        "context",
        [None, {}, {CONTEXT_TOOL_DECISIONS: "bad"}, {CONTEXT_TOOL_DECISIONS: []}],
    )
    def test_missing_or_malformed_is_empty(self, context):
        assert tool_decisions_from_context(context) == {}

    def test_skips_malformed_entries(self):
        context = {CONTEXT_TOOL_DECISIONS: {"t1": "yes", "t2": {"approved": True}}}
        assert list(tool_decisions_from_context(context)) == ["t2"]


def test_terminal_event_types():
    assert TERMINAL_EVENT_TYPES == {EVENT_COMPLETE, EVENT_ERROR, EVENT_PAUSED}


class TestBaseDefaults:
    def test_existing_executors_do_not_support_approval(self):
        assert AgentExecutorBase.supports_tool_approval is False
        assert EchoAgentExecutor.supports_tool_approval is False

    def test_default_bind_returns_self(self):
        executor = EchoAgentExecutor()
        assert executor.bind(ExecutorBinding(agent_name="a")) is executor

    def test_binding_defaults(self):
        binding = ExecutorBinding(agent_name="a")
        assert binding.tools == ()
        assert binding.before_tool_call == ()
        assert binding.system_prompt is None
        assert binding.session_store is None
