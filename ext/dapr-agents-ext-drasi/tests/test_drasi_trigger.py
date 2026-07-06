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

"""Test suite for the Drasi pub/sub trigger extension."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Optional
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.schemas import TriggerAction
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.types.exceptions import PubSubNotAvailableError
from dapr_agents.workflow.runners.agent import AgentRunner

try:
    from dapr_agents.ext.drasi import DrasiOperation, drasi_trigger
    from dapr_agents.ext.drasi.activations import (
        _DRASI_TRIGGER_DEFAULT_TASK,
        _DRASI_TRIGGER_DEFAULT_TOPIC_PREFIX,
    )

    DRASI_AVAILABLE = True
except ImportError:
    DRASI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DRASI_AVAILABLE,
    reason=(
        "dapr-agents-ext-drasi is not available. "
        "To run these tests, install the extension with: "
        "`uv sync --group test --extra drasi`",
    ),
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _safe_json_loads(data: Any) -> Any:
    """Safely parses a JSON-serializable value, returning the parsed data."""
    if isinstance(data, (dict, list)):
        return data
    return json.loads(data)


def _make_cloudevent(data: dict[str, Any], **fields) -> dict[str, Any]:
    """Build a dict that `extract_cloudevent_data` will treat as a CloudEvent."""
    base = {
        "id": "1",
        "data": data,
        "datacontenttype": "application/json",
        "pubsubname": "testpubsub",
        "source": "testpublisher",
        "specversion": "1.0",
        "topic": "testtopic",
        "type": "com.dapr.event.sent",
    }
    base.update(fields)
    return base


def _get_attr_from_wf_input(kwargs: dict[str, Any], name: str) -> str | None:
    """Return an attribute from the input to the workflow scheduler method call."""
    return _safe_json_loads(kwargs.get("input", {})).get(name)


def _get_attr_from_wf_input_metadata(kwargs: dict[str, Any], name: str) -> str | None:
    """Return a CloudEvent attribute from the input to the workflow scheduler method call."""
    return (
        _safe_json_loads(kwargs.get("input", {})).get("_message_metadata", {}).get(name)
    )


# TODO: tests may still be flaky with this workaround, will need to be replaced
async def _wait_for_completion() -> None:
    """
    Short sleep to allow background workflow scheduling to complete.
    Call this after runner entrypoint methods (`subscribe()`/`register_routes()`/`serve()`) and before assertions.
    """
    await asyncio.sleep(0.2)


@contextmanager
def _runner_raises_exception_with_cause(
    type: type[BaseException], match: str | re.Pattern[str] | None = None
):
    """
    Context manager for the runner mirroring `pytest.raises`.

    Asserts that a caught exception has a cause of the expected type
    and contains the expected message/matches the expected pattern.
    """

    # Runner always re-raises a caught exception as a RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        yield exc_info

    base_exc = exc_info.value.__cause__

    assert base_exc is not None
    assert isinstance(base_exc, type)

    if match is not None:
        pattern = (
            match if isinstance(match, re.Pattern) else re.compile(match, re.IGNORECASE)
        )
        assert pattern.search(str(base_exc)) is not None


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


def _create_mock_dapr_client(
    pubsub_names: list[str] | None = None,
    event_stream: list[Any] | None = None,
) -> MagicMock:
    """
    Create a mock DaprClient with specified pubsub components registered
    and a fixed event stream for subscriptions to consume.

    Args:
        pubsub_names: List of pubsub component names to register in the mock.
        event_stream: A list of events that a mock subscription will consume.

    Returns:
        A MagicMock configured to return the pubsub components in get_metadata
        and subscriptions that consume the given event stream.
    """
    pubsub_names = pubsub_names or []
    event_stream = event_stream or []

    mock_client = MagicMock()
    mock_sub = MagicMock()
    mock_sub.__iter__.return_value = iter(event_stream)
    mock_client.subscribe.return_value = mock_sub

    # Set up get_metadata to return the pubsub components
    mock_metadata = MagicMock()
    components = []

    for name in pubsub_names:
        component = MagicMock()
        component.type = "pubsub.redis"
        component.name = name
        components.append(component)

    mock_metadata.registered_components = components
    mock_client.get_metadata.return_value = mock_metadata

    # Support context manager usage (with DaprClient() as client:)
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    # Support constructor usage (client = DaprClient())
    mock_client.return_value = mock_client

    return mock_client


class _MockRetryPolicy:
    """Stand-in class for `dapr.ext.workflow.RetryPolicy`."""

    def __init__(
        self,
        max_number_of_attempts=1,
        first_retry_interval=timedelta(seconds=1),
        max_retry_interval=timedelta(seconds=60),
        backoff_coefficient=2.0,
        retry_timeout: Optional[timedelta] = None,
    ):
        self.max_number_of_attempts = max_number_of_attempts


class _MockTopicEventResponse:
    """
    Stand-in class for `dapr.clients.grpc._response.TopicEventResponse`.

    `conftest.py` mocks the `dapr.clients.grpc._response` module;
    patching `TopicEventResponse` prevents `_normalize_status` in `subscription.py`
    from receiving a mock status object it can't coerce, which would lead to an infinite retry loop.
    """

    def __init__(self, status: str):
        self.status = status


class _MockAgentMetadata:
    """Stand-in class for `dapr_agents.agents.config.AgentMetadata`."""

    def __init__(self, *args, **kwargs):
        pass


class _MockAgentMetadataSchema:
    """Stand-in class for `dapr_agents.agents.config.AgentMetadataSchema`."""

    def __init__(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Provide an API key and a mock Dapr client factory."""
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.base.default_dapr_client_factory",
        lambda: _create_mock_dapr_client(),
    )
    yield
    os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture(autouse=True)
def patch_subscription(monkeypatch):
    """Mock the subscription topic response to prevent issues with `_normalize_status` coercion."""
    monkeypatch.setattr(
        "dapr_agents.workflow.utils.subscription.TopicEventResponse",
        _MockTopicEventResponse,
    )


@pytest.fixture(autouse=True)
def patch_dapr_workflow_runtime(monkeypatch):
    """Mock the workflow runtime so no live Dapr instance is required."""
    import dapr.ext.workflow as wf

    mock_runtime = Mock(spec=wf.WorkflowRuntime)
    monkeypatch.setattr(wf, "WorkflowRuntime", lambda: mock_runtime)

    monkeypatch.setattr(wf, "RetryPolicy", _MockRetryPolicy)


@pytest.fixture(autouse=True)
def patch_agent_bootstrap_dapr_client(monkeypatch):
    """Mock the Dapr client used by agent bootstrapping so no live Dapr instance is required."""
    monkeypatch.setattr(
        "dapr_agents.agents.base.DaprClient",
        lambda *args, **kwargs: _create_mock_dapr_client(),
    )


@pytest.fixture(autouse=True)
def patch_agent_bootstrap_metadata(monkeypatch):
    """Mock agent metadata classes used by agent bootstrapping to avoid schema validation issues."""
    monkeypatch.setattr(
        "dapr_agents.agents.base.AgentMetadata",
        _MockAgentMetadata,
    )
    monkeypatch.setattr(
        "dapr_agents.agents.base.AgentMetadataSchema",
        _MockAgentMetadataSchema,
    )


@pytest.fixture(autouse=True)
def stub_agent_lifecycle(monkeypatch):
    """Stub agent start/stop methods to isolate the extension."""
    monkeypatch.setattr(DurableAgent, "start", Mock())
    monkeypatch.setattr(DurableAgent, "stop", Mock())


@pytest.fixture(autouse=True)
def stub_agent_route_wiring(monkeypatch):
    """Stub pub/sub + HTTP route wiring to ensure only extension routes are active."""
    monkeypatch.setattr(
        "dapr_agents.workflow.runners.agent.register_message_routes",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        "dapr_agents.workflow.runners.agent.register_http_routes",
        Mock(return_value=[]),
    )


@pytest.fixture
def setup_deps():
    """
    Return idempotent factory functions that return `DurableAgent` and `AgentRunner` instances,
    and a handle to the mock workflow client's workflow scheduler method."""

    agent: DurableAgent | None = None
    runner: AgentRunner | None = None

    # Stub workflow scheduling; we only care about the calls and inputs
    wf_client = MagicMock()
    wf_client.schedule_new_workflow.return_value = "instance-1"

    def make_agent(
        pubsub_name: str | None = None,
        topic: str | None = None,
        name: str = "TestAgent",
    ) -> DurableAgent:
        nonlocal agent

        if agent is not None:
            return agent

        llm = Mock(spec=OpenAIChatClient)
        llm.prompt_template = None
        llm.__class__.__name__ = "MockLLMClient"
        llm.provider = "MockOpenAIProvider"
        llm.api = "MockOpenAIAPI"
        llm.model = "gpt-4o-mock"

        # Allow pub/sub config to be omitted in tests
        if not pubsub_name:
            pubsub = None
        else:
            pubsub = AgentPubSubConfig(
                pubsub_name=pubsub_name or "testpubsub",
                agent_topic=topic or "testtopic",
                broadcast_topic=f"{topic}.broadcast"
                if topic is not None
                else "testtopic.broadcast",
            )

        agent = DurableAgent(
            name=name,
            role="Test Assistant",
            goal="Help with testing",
            llm=llm,
            pubsub=pubsub,
            state=AgentStateConfig(
                store=StateStoreService(store_name="teststatestore")
            ),
            execution=AgentExecutionConfig(max_iterations=5),
        )

        return agent

    def make_runner(
        pubsub_names: list[str] | None, event_stream: list[Any] | None
    ) -> AgentRunner:
        nonlocal runner

        if runner is not None:
            return runner

        runner = AgentRunner(
            wf_client=wf_client,
            client_factory=Mock(
                side_effect=lambda: _create_mock_dapr_client(pubsub_names, event_stream)
            ),
        )

        # Stub `serve()` mounts
        runner._wire_http_routes = Mock()
        runner._mount_service_routes = Mock()
        runner._mount_hitl_routes = Mock()

        return runner

    try:
        yield make_agent, make_runner, wf_client.schedule_new_workflow
    finally:
        if runner is not None:
            if agent is not None:
                runner.shutdown(agent)
            else:
                runner.shutdown()
        if agent is not None:
            # Shut down the agent explicitly in case the agent was never hosted by the runner
            # Idempotent so it's safe to call multiple times
            agent.stop()


# ---------------------------------------------------------------------------
# Agent workflow trigger behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drasi_trigger_uses_pubsub_under_subscribe(setup_deps):
    """Test that the Drasi pub/sub trigger wires pub/sub routes using the runner's `subscribe()` entrypoint."""
    query_id = "ordersquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "orderspubsub"
    drasi_topic = "orders"
    events = [
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 111,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 111,
                    },
                    "after": {
                        "orderId": "1",
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 123,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 123,
                    },
                    "after": {
                        "orderId": "2",
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 2

    # Ensure that order is preserved
    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    # Ensure that task strings are correctly generated
    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_uses_pubsub_under_register_routes(setup_deps):
    """Test that the Drasi pub/sub trigger wires pub/sub routes using the runner's `register_routes()` entrypoint."""
    query_id = "incidentsquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "incidentspubsub"
    drasi_topic = "incidents"
    events = [
        _make_cloudevent(
            data={
                "op": "d",
                "ts_ms": 456,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 456,
                    },
                    "after": {
                        "incidentId": "3",
                        "severity": 1,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "d",
                "ts_ms": 789,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 789,
                    },
                    "after": {
                        "incidentId": "4",
                        "severity": 0,
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )
    runner.register_routes(agent, fastapi_app=FastAPI())

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 2

    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_uses_pubsub_under_serve(setup_deps):
    """Test that the Drasi pub/sub trigger wires pub/sub routes using the runner's `serve()` entrypoint."""
    query_id = "potentialfraudquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "alertspubsub"
    drasi_topic = "important"
    events = [
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 67,
                "seq": 22,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 67,
                    },
                    "after": {
                        "name": "your_name",
                    },
                },
            },
            id="22",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 223,
                "seq": 33,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 223,
                    },
                    "before": {
                        "name": "your_name",
                    },
                    "after": {
                        "name": "YOUR_NAME",
                    },
                },
            },
            id="33",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )
    runner.serve(agent, app=FastAPI())

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 2

    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_uses_pubsub_independent_of_agent_pubsub(setup_deps):
    """Test that the Drasi pub/sub trigger wires pub/sub routes when the agent's pub/sub configuration is missing."""
    query_id = "gamestatequery"
    drasi_pubsub_name = "gamestatepubsub"
    drasi_topic = "gamestate"
    events = [
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 404,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 404,
                    },
                    "after": {
                        "player1": {
                            "x": 1,
                            "y": 2,
                            "z": 3,
                        },
                        "player2": {
                            "x": 4,
                            "y": 5,
                            "z": 6,
                        },
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 404,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 404,
                    },
                    "after": {
                        "player1": {
                            "x": 7,
                            "y": 8,
                            "z": 9,
                        },
                        "player2": {
                            "x": 7,
                            "y": 8,
                            "z": 9,
                        },
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent()
    runner = make_runner(pubsub_names=[drasi_pubsub_name], event_stream=events)

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 2

    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_defaults_to_agent_pubsub_component(setup_deps):
    """Test that the Drasi pub/sub trigger uses the agent's pub/sub component when the pub/sub component is omitted."""
    query_id = "searchquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_topic = "searches"
    events = [
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 911,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 911,
                    },
                    "after": {
                        "searchText": "what is bozosort",
                    },
                },
            },
            id="1",
            pubsubname=agent_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 999,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 999,
                    },
                    "after": {
                        "searchText": "how to delete search history",
                    },
                },
            },
            id="2",
            pubsubname=agent_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    drasi_trigger(
        agent,
        query_id=query_id,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 2

    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_defaults_to_derived_topic(setup_deps):
    """Test that the Drasi pub/sub trigger uses the query ID to derive the pub/sub topic when the topic is omitted."""
    query_id = "goalsquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "goalspubsub"
    drasi_topic = f"{_DRASI_TRIGGER_DEFAULT_TOPIC_PREFIX}-{query_id}"
    events = [
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 1260000,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 1260000,
                    },
                    "before": {
                        "countryCode": "BA",
                        "goals": 0,
                    },
                    "after": {
                        "countryCode": "BA",
                        "goals": 1,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 4680000,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 4680000,
                    },
                    "before": {
                        "countryCode": "CA",
                        "goals": 0,
                    },
                    "after": {
                        "countryCode": "CA",
                        "goals": 1,
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 2

    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_defaults_to_passthrough_task(setup_deps, caplog):
    """Test that the Drasi pub/sub trigger uses the default pass-through task when the task mapper is omitted."""
    query_id = "passwordupdatequery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "passwordpubsub"
    drasi_topic = "passwordtopic"
    events = [
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 401,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 401,
                    },
                    "before": {
                        "id": 1,
                        "password": "password",
                    },
                    "after": {
                        "id": 1,
                        "password": "password123",
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 406,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 406,
                    },
                    "before": {
                        "id": 2,
                        "password": "password",
                    },
                    "after": {
                        "id": 2,
                        "password": "password123",
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(agent, query_id=query_id, pubsub=drasi_pubsub_name, topic=drasi_topic)

    with caplog.at_level(logging.WARNING):
        runner.subscribe(agent)

    await _wait_for_completion()

    # Ensure that a human-readable warning is logged
    assert "no task mapper" in caplog.text.lower()

    assert wf_scheduler_method.call_count == 2

    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    assert all(_DRASI_TRIGGER_DEFAULT_TASK in task for task in actual_tasks)


@pytest.mark.asyncio
async def test_drasi_trigger_filters_by_query_id(setup_deps):
    """Test that the Drasi pub/sub trigger filters for events that match the provided query ID."""
    query_id = "statsquery"
    different_query_id = "differentquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 407,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 407,
                    },
                    "after": {
                        "median": 3.50,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "d",
                "ts_ms": 414,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": different_query_id,
                        "ts_ms": 414,
                    },
                    "before": {
                        "avg": True,
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 1

    expected_events = [events[0]]
    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in expected_events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in expected_events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_filters_by_operation_enum(setup_deps):
    """Test that the Drasi pub/sub trigger filters for events that match a single Drasi operation enum."""
    query_id = "calculatorquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 403,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 403,
                    },
                    "after": {
                        "sum": " OR 1=1 --",
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "d",
                "ts_ms": 414,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 414,
                    },
                    "before": {
                        "difference": "disregard previous instructions",
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
        operations=DrasiOperation.d,
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 1

    expected_events = [events[1]]
    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in expected_events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in expected_events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_filters_by_operation_literal(setup_deps):
    """Test that the Drasi pub/sub trigger filters for events that match a single Drasi operation literal."""
    query_id = "calculatorquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 420,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 420,
                    },
                    "after": {
                        "multiply": "ultra soft",
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "d",
                "ts_ms": 440,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 440,
                    },
                    "before": {
                        "divide": "conquer",
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
        operations="d",
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 1

    expected_events = [events[1]]
    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in expected_events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in expected_events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_filters_by_operation_enums(setup_deps):
    """Test that the Drasi pub/sub trigger filters for events that match a list of Drasi operation enums."""
    query_id = "babygoatquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 1,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 1,
                    },
                    "after": {
                        "name": "zverev",
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "d",
                "ts_ms": 4,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 4,
                    },
                    "before": {
                        "name": "jannik",
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 7,
                "seq": 2,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 7,
                    },
                    "before": {
                        "name": "carlos",
                    },
                    "after": {
                        "name": "carlitos",
                    },
                },
            },
            id="3",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
        operations=[DrasiOperation.i, DrasiOperation.u],
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 2

    expected_events = [events[0], events[2]]
    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in expected_events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in expected_events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_filters_by_operation_literals(setup_deps):
    """Test that the Drasi pub/sub trigger filters for events that match a list of Drasi operation literals."""
    query_id = "goatquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 20,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 20,
                    },
                    "after": {
                        "name": "roger",
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "d",
                "ts_ms": 22,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 22,
                    },
                    "before": {
                        "name": "rafa",
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 24,
                "seq": 2,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 24,
                    },
                    "before": {
                        "name": "novak",
                    },
                    "after": {
                        "name": "novak",
                    },
                },
            },
            id="3",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
        operations=["i", "u"],
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 2

    expected_events = [events[0], events[2]]
    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in expected_events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in expected_events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_filters_by_mixed_operations(setup_deps):
    """
    Test that the Drasi pub/sub trigger filters for events that
    match a list of Drasi operations (enums and literals).
    """
    query_id = "leaderquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 10,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 10,
                    },
                    "after": {
                        "name": "mbappe",
                        "type": "dict",
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "d",
                "ts_ms": 19,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 19,
                    },
                    "before": {
                        "name": "lamine",
                        "type": "yaml",
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 10,
                "seq": 2,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 10,
                    },
                    "before": {"name": "neymar", "type": "gif"},
                    "after": {"name": "neymar", "type": "jif"},
                },
            },
            id="3",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
        operations=[DrasiOperation.i, "u"],
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 2

    expected_events = [events[0], events[2]]
    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in expected_events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in expected_events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_filters_by_change_model(setup_deps):
    """Test that the Drasi pub/sub trigger validates the change data in events (and implicitly filters)."""
    query_id = "nucksquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 1982,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 1982,
                    },
                    "before": {
                        "count": 0,
                    },
                    "after": {
                        "count": 0,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 1994,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 1994,
                    },
                    "before": {
                        "count": 0,
                    },
                    "after": {
                        "count": 0,
                    },
                },
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 2011,
                "seq": 2,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 2011,
                    },
                    "before": {
                        "count": 0,
                    },
                    "after": {
                        "count": 0,
                    },
                },
            },
            id="3",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 2034,
                "seq": 3,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 2034,
                    },
                    "before": {
                        "period": 3,
                        "time": "19:58",
                        "home": "VAN",
                        "homeScore": 1,
                        "away": "BUF",
                        "awayScore": 0,
                        "worldEnd": False,
                        "worldEndReason": "",
                    },
                    "after": {
                        "period": 3,
                        "time": "19:59",
                        "home": "VAN",
                        "homeScore": 1,
                        "away": "BUF",
                        "awayScore": 0,
                        "worldEnd": True,
                        "worldEndReason": "divine intervention",
                    },
                },
            },
            id="4",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    class Counter(BaseModel):
        count: int

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
        change_model=Counter,
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    assert wf_scheduler_method.call_count == 3

    expected_events = [events[0], events[1], events[2]]
    cloudevent_ids = [
        _get_attr_from_wf_input_metadata(c.kwargs, "id")
        for c in wf_scheduler_method.call_args_list
    ]
    assert cloudevent_ids == [e.get("id") for e in expected_events]

    actual_tasks = [
        _get_attr_from_wf_input(c.kwargs, "task")
        for c in wf_scheduler_method.call_args_list
    ]
    expected_tasks = [f"{e.get('data', {}).get('seq')}" for e in expected_events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
async def test_drasi_trigger_ignores_events_without_change_data(setup_deps, caplog):
    """Test that the Drasi pub/sub trigger ignores events that do not contain change data."""
    query_id = "phoquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 0,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )

    with caplog.at_level(logging.WARNING):
        runner.subscribe(agent)

    await _wait_for_completion()

    # Ensure that a human-readable warning is logged
    assert "no change data" in caplog.text.lower()

    assert wf_scheduler_method.call_count == 0


@pytest.mark.asyncio
async def test_drasi_trigger_ignores_malformed_events(setup_deps):
    """Test that the Drasi pub/sub trigger ignores events that do not conform to the expected format."""
    query_id = "boringquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "foo": "bar",
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
        _make_cloudevent(
            data={
                "op": "i",
                "ts_ms": 123,
                "seq": 1,
                "payload": {},
            },
            id="2",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, wf_scheduler_method = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )
    runner.subscribe(agent)

    await _wait_for_completion()

    # Should gracefully handle malformed events
    assert wf_scheduler_method.call_count == 0


@pytest.mark.asyncio
async def test_drasi_trigger_raises_when_pubsub_is_missing(setup_deps):
    """
    Test that the Drasi pub/sub trigger fails when no pub/sub component is provided
    (as an argument or on the agent).
    """
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    drasi_pubsub_name = "missingpubsub"
    drasi_topic = "testtopic"
    events = [
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "id": 0,
                    },
                    "after": {
                        "id": 1,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, _ = setup_deps
    agent = make_agent()
    runner = make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    drasi_trigger(
        agent,
        query_id=query_id,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )

    with _runner_raises_exception_with_cause(RuntimeError, match=".*no pub/sub"):
        runner.subscribe(agent)


@pytest.mark.asyncio
async def test_drasi_trigger_raises_when_pubsub_is_not_registered(setup_deps):
    """Test that the Drasi pub/sub trigger fails when the given pub/sub component is not registered."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "missingpubsub"
    drasi_topic = "testtopic"
    events = [
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "id": 0,
                    },
                    "after": {
                        "id": 1,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, _ = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )

    # Ensure that the activation doesn't try to fall back to the agent's pub/sub component
    with _runner_raises_exception_with_cause(
        PubSubNotAvailableError,
        match=f"component.*{drasi_pubsub_name}.*topic.*{drasi_topic}",
    ):
        runner.subscribe(agent)


@pytest.mark.asyncio
async def test_drasi_trigger_raises_when_pubsub_matches_agent_pubsub(setup_deps):
    """Test that the Drasi pub/sub trigger fails when the agent pub/sub component is used and pub/sub topic matches the agent's pub/sub topic."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = agent_pubsub_name
    drasi_topic = agent_topic
    events = [
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "id": 0,
                    },
                    "after": {
                        "id": 1,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, _ = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
    )

    with _runner_raises_exception_with_cause(
        RuntimeError, match=f"{drasi_pubsub_name}.*{drasi_topic}.*matches"
    ):
        runner.subscribe(agent)


@pytest.mark.asyncio
async def test_drasi_trigger_raises_with_unsupported_operation(setup_deps):
    """Test that the Drasi pub/sub trigger fails when the provided operation is not supported."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "x",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "kind": "fatal",
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, _ = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    operation = "x"

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
        operations=operation,
    )

    # Should fail when the agent is hosted so it doesn't try to process a valid but unsupported operation
    with _runner_raises_exception_with_cause(
        TypeError, match=".*unsupported operation type"
    ):
        runner.subscribe(agent)


@pytest.mark.asyncio
async def test_drasi_trigger_raises_with_partially_valid_operation_list(setup_deps):
    """Test that the Drasi pub/sub trigger fails when the some of the provided operations are not supported."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "x",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "kind": "not really",
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, _ = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    operations = [DrasiOperation.u, "x", "i", 7]

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
        operations=operations,
    )

    with _runner_raises_exception_with_cause(
        TypeError, match=".*unsupported operation type"
    ):
        runner.subscribe(agent)


@pytest.mark.asyncio
async def test_drasi_trigger_raises_with_unsupported_change_model(setup_deps):
    """Test that the Drasi pub/sub trigger fails when the provided change model is not supported."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "id": 0,
                    },
                    "after": {
                        "id": 1,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, _ = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    change_model = 17

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=lambda event, msg_ctx: TriggerAction(task=f"{event.seq}"),
        change_model=change_model,
    )

    with _runner_raises_exception_with_cause(
        TypeError, match=".*unsupported change model type"
    ):
        runner.subscribe(agent)


@pytest.mark.asyncio
async def test_drasi_trigger_raises_with_async_task_mapper(setup_deps):
    """Test that the Drasi pub/sub trigger fails when the provided task mapper is async."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "id": 0,
                    },
                    "after": {
                        "id": 1,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, _ = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    async def async_task_mapper(event, msg_ctx):
        return TriggerAction(task=f"{event.seq}")

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=async_task_mapper,
    )

    with _runner_raises_exception_with_cause(TypeError, match="mapper.*synchronous"):
        runner.subscribe(agent)


@pytest.mark.asyncio
async def test_drasi_trigger_raises_with_non_callable_task_mapper(setup_deps):
    """Test that the Drasi pub/sub trigger fails when the provided task mapper is not callable."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "id": 0,
                    },
                    "after": {
                        "id": 1,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, _ = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    task_mapper = "nobody will ever get this far into the test suite to read this"

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=task_mapper,
    )

    with _runner_raises_exception_with_cause(TypeError, match="mapper.*callable"):
        runner.subscribe(agent)


@pytest.mark.asyncio
async def test_drasi_trigger_raises_with_async_callable_task_mapper(setup_deps):
    """
    Test that the Drasi pub/sub trigger fails when the provided task mapper is a callable object
    with an `async def` `__call__` method.
    """
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        _make_cloudevent(
            data={
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "id": 0,
                    },
                    "after": {
                        "id": 1,
                    },
                },
            },
            id="1",
            pubsubname=drasi_pubsub_name,
            topic=drasi_topic,
        ),
    ]

    make_agent, make_runner, _ = setup_deps
    agent = make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    class AsyncCallableFilter:
        async def __call__(self, event, msg_ctx):
            return TriggerAction(task=f"{event.seq}")

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        task_mapper=AsyncCallableFilter(),
    )

    with _runner_raises_exception_with_cause(TypeError, match="mapper.*synchronous"):
        runner.subscribe(agent)
