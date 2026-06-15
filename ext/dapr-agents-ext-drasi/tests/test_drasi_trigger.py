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

import json
import os
from datetime import timedelta
from typing import Any, Optional
from unittest.mock import MagicMock, Mock

from fastapi import FastAPI
from pydantic import BaseModel
import pytest

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
)
from dapr_agents.agents.schemas import TriggerAction
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners.agent import AgentRunner
from dapr_agents.ext.drasi import (  # type: ignore[import-not-found]
    DRASI_TRIGGER_DEFAULT_TASK,
    DRASI_TRIGGER_DEFAULT_TOPIC_PREFIX,
    drasi_trigger,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers (mirror tests/workflow/test_activation_hooks.py)
# ---------------------------------------------------------------------------


def _safe_json_loads(data: Any) -> Any:
    """Safely parses a JSON-serializable value, returning the parsed data."""
    if isinstance(data, (dict, list)):
        return data
    return json.loads(data)


def _create_mock_dapr_client(
    pubsub_names: list[str] | None = None,
    event_stream: list[Any] | None = None,
) -> MagicMock:
    """
    Create a mock DaprClient with specified pubsub components registered and a fixed event stream for subscriptions to consume.

    Args:
        pubsub_names: List of pubsub component names to register in the mock.
        event_stream: A list of events that a mock subscription will consume.

    Returns:
        A MagicMock configured to return the pubsub components in get_metadata and subscriptions that consume the given event stream.
    """
    pubsub_names = pubsub_names or []
    event_stream = event_stream or []

    mock_client = MagicMock()
    mock_sub = MagicMock()
    mock_sub.__iter__.return_value = iter(event_stream)
    mock_client.subscribe.return_value = mock_sub

    # Support ``subscribe_with_handler`` which returns a no-op closer and immediately consumes a fixed event stream when the agent is hosted
    def mock_sub_with_handler_fn(*args, **kwargs):
        for message in event_stream:
            # TODO: ideally want to avoid hardcoding here, support positional args too?
            handler_fn = kwargs.get("handler_fn")
            if handler_fn:
                handler_fn(message)

        return lambda: None

    mock_sub_with_handler = MagicMock()
    mock_sub_with_handler.side_effect = mock_sub_with_handler_fn
    mock_client.subscribe_with_handler = mock_sub_with_handler

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


@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
    """Mock the workflow runtime so no live Dapr instance is required."""
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
        ):
            self.max_number_of_attempts = max_number_of_attempts

    monkeypatch.setattr(wf, "RetryPolicy", MockRetryPolicy)
    yield mock_runtime


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Provide an API key and a mock Dapr client factory."""
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.base.default_dapr_client_factory",
        Mock(side_effect=lambda: _create_mock_dapr_client()),
    )
    yield
    os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture(autouse=True)
def stub_agent_lifecycle(monkeypatch):
    """Stub agent start/stop (they touch Dapr) so tests isolate activation."""
    monkeypatch.setattr(DurableAgent, "start", Mock())
    monkeypatch.setattr(DurableAgent, "stop", Mock())


@pytest.fixture(autouse=True)
def stub_route_wiring(monkeypatch):
    """Stub pub/sub + HTTP route wiring so host methods isolate activation."""
    monkeypatch.setattr(
        "dapr_agents.workflow.runners.agent.register_message_routes",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        "dapr_agents.workflow.runners.agent.register_http_routes",
        Mock(return_value=[]),
    )


def _make_agent(
    pubsub_name: str | None = None,
    topic: str | None = None,
    name: str = "TestAgent",
) -> DurableAgent:
    llm = Mock(spec=OpenAIChatClient)
    llm.prompt_template = None
    llm.__class__.__name__ = "MockLLMClient"
    llm.provider = "MockOpenAIProvider"
    llm.api = "MockOpenAIAPI"
    llm.model = "gpt-4o-mock"

    # Allow pubsub config to be omitted in tests
    if not pubsub_name and not topic:
        pubsub = None
    else:
        pubsub = AgentPubSubConfig(
            pubsub_name=pubsub_name or "testpubsub",
            agent_topic=topic or "testtopic",
            broadcast_topic=f"{topic}.broadcast"
            if topic is not None
            else "testtopic.broadcast",
        )

    return DurableAgent(
        name=name,
        role="Test Assistant",
        goal="Help with testing",
        llm=llm,
        pubsub=pubsub,
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
        execution=AgentExecutionConfig(max_iterations=5),
    )


def _make_runner(
    pubsub_names: list[str],
    event_stream: list[Any],
) -> AgentRunner:
    runner = AgentRunner(
        wf_client=MagicMock(),
        client_factory=Mock(
            side_effect=lambda: _create_mock_dapr_client(pubsub_names, event_stream)
        ),
    )

    # Stub serve() mounts
    runner._wire_http_routes = Mock()  # type: ignore[method-assign]
    runner._mount_service_routes = Mock()  # type: ignore[method-assign]
    runner._mount_hitl_routes = Mock()  # type: ignore[method-assign]

    # Stub workflow scheduling since we only care about the calls themselves
    runner.run_sync = Mock(return_value="instance-1")  # type: ignore[method-assign]

    return runner


# ---------------------------------------------------------------------------
# Agent workflow trigger behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_uses_pubsub_under_subscribe():
    """Test that the Drasi extension wires pub/sub routes correctly using the `subscribe()` method."""
    query_id = "orders-query"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "orderspubsub"
    drasi_topic = "orders"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
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
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "stock-notifications-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
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
            "id": "2",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "stock-notifications-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "orders"

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        mapper=lambda _: TriggerAction(task=task_str),
    )
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    # Ensure order is preserved
    cloudevent_ids = [
        _safe_json_loads(c.kwargs.get("payload", {}))
        .get("_message_metadata", {})
        .get("id")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    # Ensure task strings are correctly generated
    actual_tasks = [
        _safe_json_loads(c.kwargs.get("payload", {})).get("task")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    expected_tasks = [task_str for _ in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_uses_pubsub_under_register_routes():
    """Test that the Drasi extension wires pub/sub routes correctly using the `register_routes()` method."""
    query_id = "incidents-query"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "incidentspubsub"
    drasi_topic = "incidents"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
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
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "stock-notifications-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
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
            "id": "2",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "incidents-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "incidents"

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        mapper=lambda _: TriggerAction(task=task_str),
    )
    runner.register_routes(agent, fastapi_app=FastAPI())

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    cloudevent_ids = [
        _safe_json_loads(c.kwargs.get("payload", {}))
        .get("_message_metadata", {})
        .get("id")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _safe_json_loads(c.kwargs.get("payload", {})).get("task")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    expected_tasks = [task_str for _ in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_uses_pubsub_under_serve():
    """Test that the Drasi extension wires pub/sub routes correctly using the `serve()` method."""
    query_id = "potential-fraud-query"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "alertspubsub"
    drasi_topic = "important"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "i",
                "ts_ms": 67,
                "seq": 22,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 67,
                    },
                    "after": {
                        "name": "your_name_here",
                    },
                },
            },
            "id": "22",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "truth-nuke-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 223,
                "seq": 33,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 223,
                    },
                    "before": {
                        "name": "your_name_here",
                    },
                    "after": {
                        "name": "YOUR_NAME_HERE",
                    },
                },
            },
            "id": "33",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "truth-nuke-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "truth"

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        mapper=lambda _: TriggerAction(task=task_str),
    )
    runner.serve(agent, app=FastAPI())

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    cloudevent_ids = [
        _safe_json_loads(c.kwargs.get("payload", {}))
        .get("_message_metadata", {})
        .get("id")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _safe_json_loads(c.kwargs.get("payload", {})).get("task")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    expected_tasks = [task_str for _ in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_uses_pubsub_independent_of_agent_pubsub():
    """Test that the Drasi extension wires pub/sub routes correctly when the agent's pub/sub configuration is missing."""
    query_id = "game-state-query"
    drasi_pubsub_name = "gamestatepubsub"
    drasi_topic = "gamestate"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
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
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "game-state-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
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
            "id": "2",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "game-state-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "gamestate"

    agent = _make_agent()
    runner = _make_runner(pubsub_names=[drasi_pubsub_name], event_stream=events)

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        mapper=lambda _: TriggerAction(task=task_str),
    )
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    cloudevent_ids = [
        _safe_json_loads(c.kwargs.get("payload", {}))
        .get("_message_metadata", {})
        .get("id")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _safe_json_loads(c.kwargs.get("payload", {})).get("task")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    expected_tasks = [task_str for _ in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_defaults_to_agent_pubsub_component():
    """Test that the Drasi extension uses the agent's pub/sub component when no pub/sub component is provided."""
    query_id = "searches-query"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_topic = "searches"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": agent_pubsub_name,
            "data": {
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
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "searches-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": agent_pubsub_name,
            "data": {
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
            "id": "2",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "searches-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "summarize"

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    drasi_trigger(
        agent,
        query_id=query_id,
        topic=drasi_topic,
        mapper=lambda _: TriggerAction(task=task_str),
    )
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    cloudevent_ids = [
        _safe_json_loads(c.kwargs.get("payload", {}))
        .get("_message_metadata", {})
        .get("id")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _safe_json_loads(c.kwargs.get("payload", {})).get("task")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    expected_tasks = [task_str for _ in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_defaults_to_derived_topic():
    """Test that the Drasi extension uses the query ID to derive the pub/sub topic when no topic is provided."""
    query_id = "goals-query"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "goalspubsub"
    drasi_topic = f"{DRASI_TRIGGER_DEFAULT_TOPIC_PREFIX}-{query_id}"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
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
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "searches-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "i",
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
            "id": "2",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "searches-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "goals"

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        mapper=lambda _: TriggerAction(task=task_str),
    )
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    cloudevent_ids = [
        _safe_json_loads(c.kwargs.get("payload", {}))
        .get("_message_metadata", {})
        .get("id")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _safe_json_loads(c.kwargs.get("payload", {})).get("task")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    expected_tasks = [task_str for _ in events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_defaults_to_passthrough_task():
    """Test that the Drasi extension uses the default pass-through task when no custom task mapping is provided."""
    query_id = "password-update-query"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "passwordpubsub"
    drasi_topic = "passwordtopic"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
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
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
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
            "id": "2",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
    ]

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(agent, query_id=query_id, pubsub=drasi_pubsub_name, topic=drasi_topic)
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    cloudevent_ids = [
        _safe_json_loads(c.kwargs.get("payload", {}))
        .get("_message_metadata", {})
        .get("id")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == [e.get("id") for e in events]

    actual_tasks = [
        _safe_json_loads(c.kwargs.get("payload", {})).get("task")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert all(DRASI_TRIGGER_DEFAULT_TASK in task for task in actual_tasks)


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_ignores_malformed_events():
    """Test that the Drasi extension ignores events that do not conform to the expected format."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "foo": "bar",
            },
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "i",
                "ts_ms": 123,
                "seq": 1,
                "payload": "abc",
            },
            "id": "2",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "ignored"

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        mapper=lambda _: TriggerAction(task=task_str),
    )
    runner.subscribe(agent)

    # Should gracefully handle malformed events
    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_filters_by_single_operation():
    """Test that the Drasi extension correctly filters events for a single Drasi operation."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "i",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "after": {
                        "count": 1,
                    },
                },
            },
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "d",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "sum": 0,
                    },
                },
            },
            "id": "2",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "test"

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        operation="d",
        mapper=lambda _: TriggerAction(task=task_str),
    )
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 1  # type: ignore[attr-defined]

    expected_events = [events[1]]
    cloudevent_ids = [
        _safe_json_loads(c.kwargs.get("payload", {}))
        .get("_message_metadata", {})
        .get("id")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == [e.get("id") for e in expected_events]

    actual_tasks = [
        _safe_json_loads(c.kwargs.get("payload", {})).get("task")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    expected_tasks = [task_str for _ in expected_events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_filters_by_multiple_operations():
    """Test that the Drasi extension correctly filters events for multiple Drasi operations."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "i",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "after": {
                        "count": 1,
                    },
                },
            },
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "d",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "sum": 0,
                    },
                },
            },
            "id": "2",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "after": {
                        "difference": 2,
                    },
                },
            },
            "id": "3",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "test"

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        operation=["i", "u"],
        mapper=lambda _: TriggerAction(task=task_str),
    )
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    expected_events = [events[0], events[2]]
    cloudevent_ids = [
        _safe_json_loads(c.kwargs.get("payload", {}))
        .get("_message_metadata", {})
        .get("id")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == [e.get("id") for e in expected_events]

    actual_tasks = [
        _safe_json_loads(c.kwargs.get("payload", {})).get("task")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    expected_tasks = [task_str for _ in expected_events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_filters_by_result_model():
    """Test that the Drasi extension correctly validates the individual changes within Drasi change events."""

    class Counter(BaseModel):
        count: int

    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "differentpubsub"
    drasi_topic = "differenttopic"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "count": 0,
                    },
                    "after": {
                        "count": 1,
                    },
                },
            },
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "d",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "count": 1,
                    },
                },
            },
            "id": "2",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "before": {
                        "sum": 0,
                    },
                    "after": {
                        "sum": 1,
                    },
                },
            },
            "id": "3",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "i",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": query_id,
                        "ts_ms": 0,
                    },
                    "after": {
                        "count": 0,
                    },
                },
            },
            "id": "4",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "test"

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        mapper=lambda _: TriggerAction(task=task_str),
        result_model=Counter,
    )
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 3  # type: ignore[attr-defined]

    expected_events = [events[0], events[1], events[3]]
    cloudevent_ids = [
        _safe_json_loads(c.kwargs.get("payload", {}))
        .get("_message_metadata", {})
        .get("id")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == [e.get("id") for e in expected_events]

    actual_tasks = [
        _safe_json_loads(c.kwargs.get("payload", {})).get("task")
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    expected_tasks = [task_str for _ in expected_events]
    assert actual_tasks == expected_tasks


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_raises_when_given_pubsub_is_not_registered():
    """Test that the Drasi extension fails when the given pub/sub component is not registered."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = "missingpubsub"
    drasi_topic = "testtopic"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
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
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "test"

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        mapper=lambda _: TriggerAction(task=task_str),
    )

    # Ensure that the activation doesn't try to fall back to the agent's pub/sub component
    with pytest.raises(RuntimeError):
        runner.subscribe(agent)


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_raises_when_pubsub_matches_agent_pubsub():
    """Test that the Drasi extension fails when the agent pub/sub component is used and pub/sub topic matches the agent's pub/sub topic."""
    query_id = "testquery"
    agent_pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_pubsub_name = agent_pubsub_name
    drasi_topic = agent_topic
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
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
            "id": "1",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
    ]
    task_str = "test"

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    drasi_trigger(
        agent,
        query_id=query_id,
        pubsub=drasi_pubsub_name,
        topic=drasi_topic,
        mapper=lambda _: TriggerAction(task=task_str),
    )

    with pytest.raises(RuntimeError):
        runner.subscribe(agent)
