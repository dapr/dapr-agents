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

"""Lifecycle initialization for direct ``AgentRunner.run_stream`` calls."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pytest
from dapr.ext.workflow import MCPToolDef
from fastapi import FastAPI

from dapr_agents.agents.configs import AgentMCPConfig
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.tool.executor import AgentToolExecutor
from dapr_agents.types import AgentError
from dapr_agents.types.activation import ActivationContext
from dapr_agents.types.streaming import AgentStreamChunk, StreamChunkType
from dapr_agents.workflow.runners.agent import AgentRunner


class _StreamConsumer:
    def __init__(
        self,
        events: list[str],
        chunks: Optional[list[AgentStreamChunk]] = None,
    ) -> None:
        self.events = events
        self.closed = False
        self._chunks = iter(chunks or [])

    async def astart(self) -> None:
        self.events.append("consumer-start")

    async def aclose(self) -> None:
        if not self.closed:
            self.events.append("consumer-close")
            self.closed = True

    def __aiter__(self) -> "_StreamConsumer":
        return self

    async def __anext__(self) -> AgentStreamChunk:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


def _make_agent_state(
    events: list[str],
    *,
    mcp_enabled: bool = True,
    server_names: Optional[list[str]] = None,
) -> DurableAgent:
    agent = object.__new__(DurableAgent)
    agent.name = "stream-lifecycle-agent"
    agent._mcp_config = AgentMCPConfig(enabled=mcp_enabled)
    agent._mcp_tools_connected = False
    agent._discovered_mcpserver_names = list(server_names or [])
    agent.tool_executor = AgentToolExecutor(tools=[])
    agent.execution = SimpleNamespace(tool_choice=None)
    agent._infra = None
    agent._activations = []
    agent._activation_window_open = True
    agent._started = False
    agent.instrumentor = None
    return agent


def _make_agent(
    events: list[str],
    *,
    mcp_enabled: bool = True,
    server_names: Optional[list[str]] = None,
) -> DurableAgent:
    agent = _make_agent_state(
        events,
        mcp_enabled=mcp_enabled,
        server_names=server_names,
    )

    def start() -> None:
        if agent._started:
            events.append("agent-start-existing")
            raise RuntimeError("already started")
        agent._started = True
        events.append("agent-start")

    def stop() -> None:
        if agent._started:
            events.append("agent-stop")
        agent._started = False

    agent.start = start
    agent.stop = stop
    return agent


def _make_borrowed_runtime_agent(
    events: list[str],
    runtime: Any,
) -> DurableAgent:
    agent = _make_agent_state(events)
    agent._runtime = runtime
    agent._runtime_owned = False
    agent._registered = True
    agent.configuration = None
    agent._infra = SimpleNamespace(registry_state=None)
    agent._hooks = None
    return agent


def _make_runner(
    events: list[str],
    *,
    on_schedule: Optional[Callable[[], None]] = None,
    consumer_chunks: Optional[list[AgentStreamChunk]] = None,
) -> tuple[AgentRunner, list[_StreamConsumer]]:
    runner = AgentRunner(
        wf_client=MagicMock(),
        client_factory=lambda: MagicMock(),
    )
    consumers: list[_StreamConsumer] = []

    def build_consumer(**_kwargs: Any) -> _StreamConsumer:
        consumer = _StreamConsumer(events, list(consumer_chunks or []))
        consumers.append(consumer)
        return consumer

    async def schedule_workflow(*_args: Any, **_kwargs: Any) -> str:
        events.append("workflow-schedule")
        if on_schedule is not None:
            on_schedule()
        return "stream-instance"

    runner._build_stream_consumer = build_consumer
    runner.discover_entry = MagicMock(return_value=lambda: None)
    runner.run_workflow_async = schedule_workflow
    return runner, consumers


async def _drain_stream(runner: AgentRunner, agent: DurableAgent) -> None:
    async for _ in runner.run_stream(
        agent,
        payload={"task": "test"},
        listener={"type": "in_process"},
    ):
        pass


async def _consume(stream: Any) -> None:
    async for _ in stream:
        pass


def _mcp_tool(server_name: str = "tool-server") -> MCPToolDef:
    return MCPToolDef(
        name="lookup",
        description="Look up a value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        server_name=server_name,
        call_tool_workflow=f"dapr.internal.mcp.{server_name}.CallTool.lookup",
    )


def _stream_chunk() -> AgentStreamChunk:
    return AgentStreamChunk(
        sequence=1,
        type=StreamChunkType.CONTENT_DELTA,
        agent="stream-lifecycle-agent",
        workflow_instance_id="stream-instance",
        turn=0,
        root_instance_id="stream-instance",
    )


def _install_mcp_client(
    monkeypatch: pytest.MonkeyPatch,
    client_type: type,
) -> None:
    module = ModuleType("dapr.ext.workflow.aio")
    module.DaprMCPClient = client_type
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_run_stream_initializes_tools_and_activation_before_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    connect_calls: list[str] = []

    class FakeDaprMCPClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.connected: list[str] = []

        async def connect(self, server_name: str) -> None:
            connect_calls.append(server_name)
            events.append(f"mcp-connect:{server_name}")
            self.connected.append(server_name)

        def get_all_tools(self) -> list[MCPToolDef]:
            return [_mcp_tool()]

        def get_connected_servers(self) -> list[str]:
            return self.connected

    _install_mcp_client(monkeypatch, FakeDaprMCPClient)

    agent = _make_agent(events, server_names=["tool-server"])
    activation_contexts: list[ActivationContext] = []
    closer_calls: list[int] = []

    def activate(context: ActivationContext) -> Callable[[], None]:
        events.append("activation")
        activation_contexts.append(context)
        return lambda: closer_calls.append(1)

    agent.add_activation(activate)

    def assert_initialized() -> None:
        assert agent.tool_executor.get_tool("lookup") is not None
        assert agent.execution.tool_choice == "auto"
        assert len(activation_contexts) == 1

    runner, consumers = _make_runner(events, on_schedule=assert_initialized)
    stream = runner.run_stream(
        agent,
        payload={"task": "test"},
        listener={"type": "in_process"},
    )

    assert events == []
    asyncio.run(_consume(stream))

    assert events == [
        "mcp-connect:tool-server",
        "agent-start",
        "activation",
        "consumer-start",
        "workflow-schedule",
        "consumer-close",
    ]
    assert activation_contexts[0].app is None
    assert connect_calls == ["tool-server"]
    assert len(consumers) == 1
    assert consumers[0].closed
    assert closer_calls == []

    asyncio.run(_drain_stream(runner, agent))

    assert connect_calls == ["tool-server"]
    assert len(activation_contexts) == 1
    assert agent.tool_executor.get_tool("lookup") is not None
    assert agent.tool_executor.get_tool_names() == ["lookup"]
    assert events[-4:] == [
        "agent-start-existing",
        "consumer-start",
        "workflow-schedule",
        "consumer-close",
    ]
    assert len(consumers) == 2
    assert all(consumer.closed for consumer in consumers)
    assert closer_calls == []

    runner.shutdown(agent)

    assert closer_calls == [1]
    assert not agent._started


def test_closing_stream_early_only_closes_stream_consumer() -> None:
    events: list[str] = []
    closer_calls: list[int] = []
    agent = _make_agent(events)
    agent.add_activation(lambda _context: lambda: closer_calls.append(1))
    runner, consumers = _make_runner(
        events,
        consumer_chunks=[_stream_chunk()],
    )

    async def consume_one_chunk() -> None:
        stream = runner.run_stream(
            agent,
            payload={"task": "test"},
            listener={"type": "in_process"},
        )
        chunk = await anext(stream)

        assert chunk.sequence == 1
        assert len(consumers) == 1
        assert not consumers[0].closed
        assert closer_calls == []
        assert agent._started

        await stream.aclose()

    asyncio.run(consume_one_chunk())

    assert consumers[0].closed
    assert closer_calls == []
    assert agent._started

    runner.shutdown(agent)

    assert closer_calls == [1]
    assert not agent._started


def test_run_stream_propagates_mcp_failure_before_hosting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FailingDaprMCPClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def connect(self, server_name: str) -> None:
            events.append(f"mcp-connect:{server_name}")
            raise ConnectionError("MCP unavailable")

        def get_all_tools(self) -> list[MCPToolDef]:
            return []

        def get_connected_servers(self) -> list[str]:
            return []

    _install_mcp_client(monkeypatch, FailingDaprMCPClient)

    agent = _make_agent(events, server_names=["tool-server"])
    activation_calls: list[int] = []
    agent.add_activation(lambda _context: activation_calls.append(1))
    runner, consumers = _make_runner(events)

    with pytest.raises(AgentError, match="Failed to connect to MCPServer"):
        asyncio.run(_drain_stream(runner, agent))

    assert events == ["mcp-connect:tool-server"]
    assert activation_calls == []
    assert consumers == []
    assert agent not in runner._managed_agents
    assert not agent._started

    runner.shutdown()


def test_failed_activation_does_not_shutdown_borrowed_runtime() -> None:
    events: list[str] = []
    runtime = MagicMock()
    runtime.start.side_effect = RuntimeError("already running")
    agent = _make_borrowed_runtime_agent(events, runtime)

    def fail_activation(_context: ActivationContext) -> None:
        raise ValueError("activation unavailable")

    agent.add_activation(fail_activation)
    runner, consumers = _make_runner(events)

    with pytest.raises(RuntimeError, match="failed during hosting"):
        asyncio.run(_drain_stream(runner, agent))

    runtime.start.assert_called_once_with()
    runtime.shutdown.assert_not_called()
    assert not agent._started
    assert agent not in runner._managed_agents
    assert id(agent) not in runner._activated_agent_ids
    assert consumers == []
    assert "workflow-schedule" not in events

    runner.shutdown()
    runtime.shutdown.assert_not_called()


def test_run_stream_rolls_back_failed_activation_and_can_retry() -> None:
    events: list[str] = []
    activation_calls: list[ActivationContext] = []
    closer_calls: list[int] = []
    agent = _make_agent(events)

    def activate(context: ActivationContext) -> Optional[Callable[[], None]]:
        activation_calls.append(context)
        events.append("activation")
        if len(activation_calls) == 1:
            raise ValueError("activation unavailable")
        return lambda: closer_calls.append(1)

    agent.add_activation(activate)
    runner, consumers = _make_runner(events)

    with pytest.raises(RuntimeError, match="failed during hosting"):
        asyncio.run(_drain_stream(runner, agent))

    assert events == ["agent-start", "activation", "agent-stop"]
    assert consumers == []
    assert agent not in runner._managed_agents
    assert id(agent) not in runner._activated_agent_ids

    asyncio.run(_drain_stream(runner, agent))

    assert len(activation_calls) == 2
    assert events[-5:] == [
        "agent-start",
        "activation",
        "consumer-start",
        "workflow-schedule",
        "consumer-close",
    ]
    assert closer_calls == []

    runner.shutdown(agent)
    assert closer_calls == [1]


@pytest.mark.parametrize(
    ("mcp_enabled", "server_names"),
    [
        (False, ["ignored-server"]),
        (True, []),
    ],
)
def test_run_stream_handles_disabled_or_empty_discovery(
    monkeypatch: pytest.MonkeyPatch,
    mcp_enabled: bool,
    server_names: list[str],
) -> None:
    events: list[str] = []

    class UnexpectedDaprMCPClient:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("MCP client should not be created")

    _install_mcp_client(monkeypatch, UnexpectedDaprMCPClient)

    agent = _make_agent(
        events,
        mcp_enabled=mcp_enabled,
        server_names=server_names,
    )
    runner, consumers = _make_runner(events)

    asyncio.run(_drain_stream(runner, agent))

    assert agent._mcp_tools_connected
    assert "workflow-schedule" in events
    assert len(consumers) == 1
    assert consumers[0].closed

    runner.shutdown(agent)


def test_stream_after_serve_keeps_original_activation_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class UnexpectedDaprMCPClient:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("MCP client should not be created")

    _install_mcp_client(monkeypatch, UnexpectedDaprMCPClient)

    agent = _make_agent(events)
    activation_contexts: list[ActivationContext] = []
    closer_calls: list[int] = []

    def activate(context: ActivationContext) -> Callable[[], None]:
        activation_contexts.append(context)
        return lambda: closer_calls.append(1)

    agent.add_activation(activate)
    runner, consumers = _make_runner(events)
    runner._wire_pubsub_routes = MagicMock()
    runner._wire_http_routes = MagicMock()
    runner._mount_service_routes = MagicMock()
    runner._mount_hitl_routes = MagicMock()
    app = FastAPI()

    assert runner.serve(agent, app=app) is app
    asyncio.run(_drain_stream(runner, agent))

    assert len(activation_contexts) == 1
    assert activation_contexts[0].app is app
    assert len(consumers) == 1
    assert consumers[0].closed
    assert closer_calls == []
    assert agent._started

    runner.shutdown(agent)

    assert closer_calls == [1]
    assert not agent._started
