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

"""Tests for the history-derived conversation path.

``test_call_llm_streaming`` holds the old path's copy; at the contract release
the old copy goes and this module takes its name.
"""

from __future__ import annotations
from typing import Any, List
from unittest.mock import Mock
import pytest
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.streaming.listeners import (
    register_stream_listener,
    _USER_LISTENERS,
)
from dapr_agents.types.streaming import StreamChunkType
from tests.streaming.test_call_llm_streaming import (  # noqa: F401
    patch_dapr_check,
    _CapturingListener,
    _stream_context,
    _content_stream,
)


@pytest.fixture(autouse=True)
def registered_capturing_listener():
    _USER_LISTENERS.pop("test-capture", None)
    captured: List[_CapturingListener] = []

    def factory(config, **kwargs):
        listener = _CapturingListener()
        captured.append(listener)
        return listener

    register_stream_listener("test-capture", factory)
    try:
        yield captured
    finally:
        _USER_LISTENERS.pop("test-capture", None)


@pytest.fixture
def mock_llm() -> Mock:
    mock = Mock(spec=OpenAIChatClient)
    mock.generate = Mock()
    mock.prompt_template = None
    mock.__class__.__name__ = "MockLLMClient"
    mock.provider = "openai"
    mock.api = "chat"
    mock.model = "gpt-4o-mock"
    return mock


@pytest.fixture
def durable_agent(mock_llm: Mock) -> DurableAgent:
    return DurableAgent(
        name="StreamingAgent",
        role="Test Assistant",
        goal="Help with streaming tests",
        instructions=["Be helpful"],
        llm=mock_llm,
        pubsub=AgentPubSubConfig(
            pubsub_name="testpubsub",
            agent_topic="StreamingAgent",
        ),
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
        registry=AgentRegistryConfig(
            store=StateStoreService(store_name="testregistry")
        ),
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="teststatestore",
                workflow_instance_id="test_session",
            )
        ),
        execution=AgentExecutionConfig(max_iterations=3, streaming=True),
    )


def _run_call_llm(
    agent: DurableAgent,
    monkeypatch: pytest.MonkeyPatch,
    response: Any,
    *,
    turn: int = 1,
) -> dict:
    """Drive the real ``call_llm`` activity for a streaming session with a given
    provider ``generate`` return value, stubbing the surrounding state I/O.

    Isolates the real-stream-vs-fallback *routing* decision (the branch keyed on
    ``_is_chunk_iterator``) from conversation rebuilding, so the test asserts
    only on what reaches the stream.
    """
    monkeypatch.setattr(
        agent.prompting_helper, "build_initial_messages", lambda *a, **k: []
    )
    monkeypatch.setattr(agent, "get_llm_tools", lambda *a, **k: [])
    monkeypatch.setattr(agent.text_formatter, "print_message", lambda *a, **k: None)
    agent.llm.generate = Mock(return_value=response)
    return agent.call_llm(
        ctx=Mock(),
        payload={
            "instance_id": "wf-1",
            "turn": turn,
            "stream_context": _stream_context(),
            "new_messages": [],
            "history_len": 0,
        },
    )


class TestCallLLMStreamingRouting:
    """End-to-end routing in ``call_llm`` for a streaming session.

    A real provider stream must go down the delta path
    (``_consume_llm_stream``); a provider fallback (what the Dapr Conversation
    API returns today) must go down the uniform START+TURN_COMPLETE envelope.
    This guards the real-delta branch from going stale while only the fallback
    exercises in practice — so there are no surprises once Dapr streams for real.
    """

    def test_real_stream_routes_to_delta_path(
        self, durable_agent: DurableAgent, monkeypatch, registered_capturing_listener
    ) -> None:
        result = _run_call_llm(
            durable_agent, monkeypatch, _content_stream(["Hello ", "world"])
        )
        assert result["role"] == "assistant"
        assert result["content"] == "Hello world"
        assert len(registered_capturing_listener) == 1
        types = [c.type for c in registered_capturing_listener[0].chunks]
        assert types[0] is StreamChunkType.START
        assert types[-1] is StreamChunkType.TURN_COMPLETE
        # Real deltas flowed — not the single-envelope fallback.
        assert types.count(StreamChunkType.CONTENT_DELTA) == 2
