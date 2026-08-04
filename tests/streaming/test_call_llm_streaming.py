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

"""Integration tests for call_llm streaming helpers on DurableAgent."""

from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional
from unittest.mock import MagicMock, Mock

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
from dapr_agents.types.message import (
    AssistantMessage,
    FunctionCallChunk,
    LLMChatCandidate,
    LLMChatCandidateChunk,
    LLMChatResponse,
    LLMChatResponseChunk,
    ToolCallChunk,
)
from dapr_agents.types.streaming import AgentStreamChunk, StreamChunkType


@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
    """Mock Dapr dependencies so DurableAgent can instantiate without a sidecar."""

    import dapr.ext.workflow as wf

    mock_runtime = Mock(spec=wf.WorkflowRuntime)
    monkeypatch.setattr(wf, "WorkflowRuntime", lambda: mock_runtime)

    class MockRetryPolicy:
        def __init__(
            self,
            max_number_of_attempts=1,
            first_retry_interval=timedelta(seconds=1),
            max_retry_interval=timedelta(seconds=60),
            backoff_coefficient=2.0,
            retry_timeout: Optional[timedelta] = None,
        ) -> None:
            self.max_number_of_attempts = max_number_of_attempts
            self.first_retry_interval = first_retry_interval
            self.max_retry_interval = max_retry_interval
            self.backoff_coefficient = backoff_coefficient
            self.retry_timeout = retry_timeout

    monkeypatch.setattr(wf, "RetryPolicy", MockRetryPolicy)

    os.environ["OPENAI_API_KEY"] = "test-key"

    class _MockDaprClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def __call__(self, *a, **kw):
            return self

        def get_state(self, *a, **kw):
            return Mock(data=None)

        def save_state(self, *a, **kw):
            return None

        def get_metadata(self):
            resp = MagicMock()
            resp.registered_components = []
            resp.application_id = "test-app"
            return resp

    mock_client = _MockDaprClient()
    monkeypatch.setattr("dapr.clients.DaprClient", lambda: mock_client)
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.base.DaprClient",
        lambda: mock_client,
    )

    yield


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


class _CapturingListener:
    def __init__(self) -> None:
        self.chunks: List[AgentStreamChunk] = []
        self.closed = False

    def emit(self, chunk: AgentStreamChunk) -> None:
        self.chunks.append(chunk)

    def close(self) -> None:
        self.closed = True


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


def _stream_context(**overrides) -> dict:
    base = {
        "root_instance_id": "root-1",
        "listener_config": {"type": "test-capture"},
        "parent_agent": None,
        "parent_instance_id": None,
        "depth": 0,
        "call_path": ["StreamingAgent"],
    }
    base.update(overrides)
    return base


def _content_stream(parts: Iterable[str]) -> List[LLMChatResponseChunk]:
    return [
        LLMChatResponseChunk(result=LLMChatCandidateChunk(role="assistant")),
        *[LLMChatResponseChunk(result=LLMChatCandidateChunk(content=p)) for p in parts],
        LLMChatResponseChunk(result=LLMChatCandidateChunk(finish_reason="stop")),
    ]


class TestConsumeLLMStream:
    def test_consume_stream_emits_deltas_and_returns_message(
        self, durable_agent: DurableAgent, registered_capturing_listener
    ) -> None:
        stream_ctx = _stream_context()
        result = durable_agent._consume_llm_stream(
            response=_content_stream(["hello ", "world"]),
            stream_ctx=stream_ctx,
            instance_id="wf-1",
            turn=1,
            phase=None,
        )
        assert result == {"role": "assistant", "content": "hello world"}
        assert len(registered_capturing_listener) == 1
        listener = registered_capturing_listener[0]
        types = [c.type for c in listener.chunks]
        assert types[0] is StreamChunkType.START
        assert types[-1] is StreamChunkType.TURN_COMPLETE
        assert types.count(StreamChunkType.CONTENT_DELTA) == 2
        assert listener.closed is True
        # TURN_COMPLETE metadata carries finish_reason
        assert listener.chunks[-1].metadata.get("finish_reason") == "stop"

    def test_tool_call_stream_reconstructs_function(
        self, durable_agent: DurableAgent, registered_capturing_listener
    ) -> None:
        stream = [
            LLMChatResponseChunk(result=LLMChatCandidateChunk(role="assistant")),
            LLMChatResponseChunk(
                result=LLMChatCandidateChunk(
                    tool_calls=[
                        ToolCallChunk(
                            index=0,
                            id="call_1",
                            type="function",
                            function=FunctionCallChunk(name="search", arguments=""),
                        )
                    ]
                )
            ),
            LLMChatResponseChunk(
                result=LLMChatCandidateChunk(
                    tool_calls=[
                        ToolCallChunk(
                            index=0,
                            function=FunctionCallChunk(arguments='{"q":"py"}'),
                        )
                    ]
                )
            ),
            LLMChatResponseChunk(
                result=LLMChatCandidateChunk(finish_reason="tool_calls")
            ),
        ]
        result = durable_agent._consume_llm_stream(
            response=stream,
            stream_ctx=_stream_context(),
            instance_id="wf-1",
            turn=1,
            phase=None,
        )
        assert result["tool_calls"][0]["function"]["name"] == "search"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"q":"py"}'
        listener = registered_capturing_listener[0]
        assert any(c.type is StreamChunkType.TOOL_CALL_DELTA for c in listener.chunks)

    def test_missing_listener_config_falls_back_to_accumulate_only(
        self, durable_agent: DurableAgent, registered_capturing_listener
    ) -> None:
        # stream_ctx without listener_config — emitter refuses to build,
        # caller still gets the accumulated message.
        ctx = _stream_context()
        ctx.pop("listener_config")
        result = durable_agent._consume_llm_stream(
            response=_content_stream(["x", "y"]),
            stream_ctx=ctx,
            instance_id="wf-1",
            turn=1,
            phase=None,
        )
        assert result == {"role": "assistant", "content": "xy"}
        assert registered_capturing_listener == []


class TestEmitNonStreamingStream:
    def test_uniform_start_complete_default_no_body(
        self, durable_agent: DurableAgent, registered_capturing_listener
    ) -> None:
        """Default: TURN_COMPLETE carries no complete_message (opt-in contract).

        Consumers fetch the final assistant message from workflow state;
        the fallback path never broadcasts it on the stream topic.
        """
        durable_agent._emit_non_streaming_stream(
            stream_ctx=_stream_context(),
            instance_id="wf-1",
            turn=1,
            phase=None,
            assistant_message={"role": "assistant", "content": "final"},
            metadata={"dapr_conversation_streaming_unsupported": True},
        )
        assert len(registered_capturing_listener) == 1
        listener = registered_capturing_listener[0]
        types = [c.type for c in listener.chunks]
        assert types == [StreamChunkType.START, StreamChunkType.TURN_COMPLETE]
        assert listener.chunks[-1].complete_message is None
        # Metadata still propagates so consumers see the fallback tag.
        assert listener.chunks[-1].metadata == {
            "dapr_conversation_streaming_unsupported": True
        }
        assert listener.closed is True

    def test_uniform_start_complete_opt_in_carries_body(
        self, durable_agent: DurableAgent, registered_capturing_listener
    ) -> None:
        ctx = _stream_context(include_complete_message=True)
        durable_agent._emit_non_streaming_stream(
            stream_ctx=ctx,
            instance_id="wf-1",
            turn=1,
            phase=None,
            assistant_message={"role": "assistant", "content": "final"},
        )
        listener = registered_capturing_listener[0]
        assert listener.chunks[-1].complete_message == {
            "role": "assistant",
            "content": "final",
        }


class TestAttributionPropagation:
    """Verify that stream_ctx fields (depth, parent, call_path) are written
    onto emitted chunks unchanged — this is the multi-agent attribution
    contract; the streaming UX depends on it rendering "agent A (via B
    via C)" correctly."""

    def test_child_stream_ctx_propagates_to_emitted_chunks(
        self, durable_agent: DurableAgent, registered_capturing_listener
    ) -> None:
        child_ctx = _stream_context(
            parent_agent="wedding-planner",
            parent_instance_id="wf-root",
            depth=1,
            call_path=["wedding-planner", "StreamingAgent"],
            trace_parent="00-trace-span-01",
        )
        durable_agent._consume_llm_stream(
            response=_content_stream(["hi"]),
            stream_ctx=child_ctx,
            instance_id="wf-child",
            turn=3,
            phase="planning",
        )
        listener = registered_capturing_listener[0]
        assert listener.chunks, "expected at least one emitted chunk"
        for chunk in listener.chunks:
            assert chunk.agent == "StreamingAgent"
            assert chunk.parent_agent == "wedding-planner"
            assert chunk.parent_instance_id == "wf-root"
            assert chunk.depth == 1
            assert chunk.call_path == ["wedding-planner", "StreamingAgent"]
            assert chunk.trace_parent == "00-trace-span-01"
            assert chunk.workflow_instance_id == "wf-child"
            assert chunk.turn == 3
            assert chunk.phase == "planning"


class TestIsChunkIterator:
    def test_llm_chat_response_is_not_iterator(self) -> None:
        from dapr_agents.types import LLMChatResponse

        resp = LLMChatResponse(results=[])
        assert DurableAgent._is_chunk_iterator(resp) is False

    def test_string_and_dict_are_not_iterators(self) -> None:
        assert DurableAgent._is_chunk_iterator("hello") is False
        assert DurableAgent._is_chunk_iterator({"a": 1}) is False

    def test_generator_is_iterator(self) -> None:
        def gen():
            yield 1

        assert DurableAgent._is_chunk_iterator(gen()) is True

    def test_list_is_iterator(self) -> None:
        assert DurableAgent._is_chunk_iterator([1, 2, 3]) is True


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
    ``_is_chunk_iterator``) from history reconstruction and persistence, so the
    test asserts only on what reaches the stream.
    """
    entry = SimpleNamespace(stream_context=_stream_context())
    monkeypatch.setattr(agent._infra, "get_state", lambda *a, **k: entry)
    monkeypatch.setattr(agent, "_reconstruct_conversation_history", lambda *a, **k: [])
    monkeypatch.setattr(
        agent.prompting_helper, "build_initial_messages", lambda *a, **k: []
    )
    monkeypatch.setattr(agent, "_sync_system_messages_with_state", lambda *a, **k: None)
    monkeypatch.setattr(agent, "get_llm_tools", lambda *a, **k: [])
    monkeypatch.setattr(agent, "_save_assistant_message", lambda *a, **k: None)
    monkeypatch.setattr(agent, "save_state", lambda *a, **k: None)
    monkeypatch.setattr(agent.text_formatter, "print_message", lambda *a, **k: None)
    agent.llm.generate = Mock(return_value=response)
    return agent.call_llm(ctx=Mock(), payload={"instance_id": "wf-1", "turn": turn})


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
        assert result == {"role": "assistant", "content": "Hello world"}
        assert len(registered_capturing_listener) == 1
        types = [c.type for c in registered_capturing_listener[0].chunks]
        assert types[0] is StreamChunkType.START
        assert types[-1] is StreamChunkType.TURN_COMPLETE
        # Real deltas flowed — not the single-envelope fallback.
        assert types.count(StreamChunkType.CONTENT_DELTA) == 2

    def test_tool_call_stream_routes_to_delta_path(
        self, durable_agent: DurableAgent, monkeypatch, registered_capturing_listener
    ) -> None:
        stream = [
            LLMChatResponseChunk(result=LLMChatCandidateChunk(role="assistant")),
            LLMChatResponseChunk(
                result=LLMChatCandidateChunk(
                    tool_calls=[
                        ToolCallChunk(
                            index=0,
                            id="call_1",
                            type="function",
                            function=FunctionCallChunk(name="search", arguments=""),
                        )
                    ]
                )
            ),
            LLMChatResponseChunk(
                result=LLMChatCandidateChunk(
                    tool_calls=[
                        ToolCallChunk(
                            index=0,
                            function=FunctionCallChunk(arguments='{"q":"py"}'),
                        )
                    ]
                )
            ),
            LLMChatResponseChunk(
                result=LLMChatCandidateChunk(finish_reason="tool_calls")
            ),
        ]
        result = _run_call_llm(durable_agent, monkeypatch, stream)
        assert result["tool_calls"][0]["function"]["name"] == "search"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"q":"py"}'
        chunks = registered_capturing_listener[0].chunks
        assert any(c.type is StreamChunkType.TOOL_CALL_DELTA for c in chunks)

    def test_dapr_fallback_routes_to_uniform_envelope(
        self, durable_agent: DurableAgent, monkeypatch, registered_capturing_listener
    ) -> None:
        fallback = LLMChatResponse(
            results=[
                LLMChatCandidate(
                    message=AssistantMessage(role="assistant", content="final answer"),
                    finish_reason="stop",
                )
            ],
            metadata={"dapr_conversation_streaming_unsupported": True},
        )
        result = _run_call_llm(durable_agent, monkeypatch, fallback)
        assert result["content"] == "final answer"
        assert len(registered_capturing_listener) == 1
        listener = registered_capturing_listener[0]
        types = [c.type for c in listener.chunks]
        # Fallback: uniform envelope, no content deltas.
        assert types == [StreamChunkType.START, StreamChunkType.TURN_COMPLETE]
        assert (
            listener.chunks[-1].metadata.get("dapr_conversation_streaming_unsupported")
            is True
        )
