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

"""Unit tests for ``dapr_agents.agents.executors.base`` and ``event``."""

import dataclasses
import inspect

import pytest

from dapr_agents.agents.executors import (
    AgentEvent,
    AgentEventType,
    AgentExecutorBase,
)
from dapr_agents.agents.executors import event as executors_event


class TestAgentEvent:
    """Covers the AgentEvent dataclass contract."""

    def test_is_frozen_dataclass(self):
        assert dataclasses.is_dataclass(AgentEvent)
        event = AgentEvent(type="complete", content="hi")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.type = "error"  # type: ignore[misc]

    def test_defaults(self):
        event = AgentEvent(type="text_delta", content="token")
        assert event.session_id is None
        assert event.metadata == {}

    def test_metadata_is_per_instance(self):
        """default_factory must give each instance its own dict."""
        a = AgentEvent(type="session", content={})
        b = AgentEvent(type="session", content={})
        a.metadata["k"] = "v"
        assert "k" not in b.metadata

    def test_accepts_all_event_types(self):
        """AgentEventType Literal must include the full RFC-defined set."""
        expected = {
            "text_delta",
            "tool_call",
            "tool_result",
            "message",
            "session",
            "complete",
            "error",
        }
        # AgentEventType is a typing.Literal; pull its arg set.
        assert set(AgentEventType.__args__) == expected  # type: ignore[attr-defined]

    def test_full_construction(self):
        event = AgentEvent(
            type="tool_call",
            content={"id": "t1", "name": "search", "arguments": {"q": "x"}},
            session_id="sess-1",
            metadata={"trace_id": "abc"},
        )
        assert event.type == "tool_call"
        assert event.session_id == "sess-1"
        assert event.metadata == {"trace_id": "abc"}


class TestAgentEventConstants:
    """Module-level EVENT_* constants are the typed mirror of AgentEventType."""

    def test_constants_cover_every_literal_value(self):
        """Each AgentEventType literal must have an EVENT_* constant."""
        literal_values = set(AgentEventType.__args__)  # type: ignore[attr-defined]
        constants = {
            name: getattr(executors_event, name)
            for name in dir(executors_event)
            if name.startswith("EVENT_")
        }
        assert set(constants.values()) == literal_values

    def test_constants_are_string_literals(self):
        assert executors_event.EVENT_TEXT_DELTA == "text_delta"
        assert executors_event.EVENT_TOOL_CALL == "tool_call"
        assert executors_event.EVENT_TOOL_RESULT == "tool_result"
        assert executors_event.EVENT_MESSAGE == "message"
        assert executors_event.EVENT_SESSION == "session"
        assert executors_event.EVENT_COMPLETE == "complete"
        assert executors_event.EVENT_ERROR == "error"


class TestAgentExecutorBase:
    """Covers the AgentExecutorBase abstract contract."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AgentExecutorBase()  # type: ignore[abstract]

    def test_run_is_abstract(self):
        assert getattr(AgentExecutorBase.run, "__isabstractmethod__", False)

    def test_get_session_is_abstract(self):
        assert getattr(AgentExecutorBase.get_session, "__isabstractmethod__", False)

    def test_run_signature(self):
        """run(prompt, *, session_id=None, context=None) per RFC #569."""
        sig = inspect.signature(AgentExecutorBase.run)
        params = sig.parameters
        assert list(params) == ["self", "prompt", "session_id", "context"]
        assert params["session_id"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["context"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["session_id"].default is None
        assert params["context"].default is None

    def test_get_session_signature(self):
        sig = inspect.signature(AgentExecutorBase.get_session)
        assert list(sig.parameters) == ["self", "session_id"]

    def test_subclass_must_implement_both_methods(self):
        class Partial(AgentExecutorBase):
            async def run(self, prompt, *, session_id=None, context=None):
                yield AgentEvent(type="complete", content="done")
                # Missing get_session -> still abstract.

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]
