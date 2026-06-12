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
from unittest.mock import MagicMock, Mock, patch

from fastapi import FastAPI
from pydantic import BaseModel
import pytest

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
)
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners.agent import AgentRunner
from dapr_agents.ext.drasi import drasi_trigger  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Fixtures and helpers (mirror tests/workflow/test_activation_hooks.py)
# ---------------------------------------------------------------------------


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
                        "queryId": "orders-query",
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
                        "queryId": "orders-query",
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

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(agent, pubsub=drasi_pubsub_name, topic=drasi_topic)
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    # Ensure order is preserved
    cloudevent_ids = [
        c.kwargs["payload"]["_message_metadata"]["id"]
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == ["1", "2"]

    # Ensure default tasks are serialized event data
    tasks = [
        json.loads(c.kwargs["payload"]["task"])
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    event_data = [e["data"] for e in events]
    assert tasks == event_data


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_uses_pubsub_under_register_routes():
    """Test that the Drasi extension wires pub/sub routes correctly using the `register_routes()` method."""
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
                        "queryId": "incidents-query",
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
                        "queryId": "incidents-query",
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

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(agent, pubsub=drasi_pubsub_name, topic=drasi_topic)
    runner.register_routes(agent, fastapi_app=FastAPI())

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    cloudevent_ids = [
        c.kwargs["payload"]["_message_metadata"]["id"]
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == ["1", "2"]

    tasks = [
        json.loads(c.kwargs["payload"]["task"])
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    event_data = [e["data"] for e in events]
    assert tasks == event_data


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_uses_pubsub_under_serve():
    """Test that the Drasi extension wires pub/sub routes correctly using the `serve()` method."""
    agent_pubsub_name = "agent-notifications"
    agent_topic = "agent-inbox"
    drasi_pubsub_name = "alerts"
    drasi_topic = "important"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": drasi_pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 67,
                "seq": 22,
                "payload": {
                    "source": {
                        "queryId": "potential-fraud-query",
                        "ts_ms": 67,
                    },
                    "before": {
                        "name": "your_name_here",
                    },
                    "after": {
                        "name": "YOUR_NAME_HERE",
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
                        "queryId": "password-update-query",
                        "ts_ms": 223,
                    },
                    "before": {
                        "userId": "1",
                        "password": "password",
                    },
                    "after": {
                        "userId": "1",
                        "password": "password1",
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

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(
        pubsub_names=[agent_pubsub_name, drasi_pubsub_name], event_stream=events
    )

    drasi_trigger(agent, pubsub=drasi_pubsub_name, topic=drasi_topic)
    runner.serve(agent, app=FastAPI())

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    cloudevent_ids = [
        c.kwargs["payload"]["_message_metadata"]["id"]
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == ["22", "33"]

    tasks = [
        json.loads(c.kwargs["payload"]["task"])
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    event_data = [e["data"] for e in events]
    assert tasks == event_data


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_uses_pubsub_independent_of_agent_pubsub():
    """Test that the Drasi extension wires pub/sub routes correctly if the agent pub/sub is missing."""
    drasi_pubsub_name = "gamepubsub"
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
                        "queryId": "game-state-query",
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
                        "queryId": "game-state-query",
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

    agent = _make_agent()
    runner = _make_runner(pubsub_names=[drasi_pubsub_name], event_stream=events)

    drasi_trigger(agent, pubsub=drasi_pubsub_name, topic=drasi_topic)
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    cloudevent_ids = [
        c.kwargs["payload"]["_message_metadata"]["id"]
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == ["1", "2"]

    tasks = [
        json.loads(c.kwargs["payload"]["task"])
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    event_data = [e["data"] for e in events]
    assert tasks == event_data


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_defaults_to_agent_pubsub_component():
    """Test that the Drasi extension uses the agent's pub/sub component when no pub/sub component is specified."""
    pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_topic = "searches"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": pubsub_name,
            "data": {
                "op": "i",
                "ts_ms": 911,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "searches-query",
                        "ts_ms": 911,
                    },
                    "after": {
                        "searchText": "questionable search text",
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
            "pubsubname": pubsub_name,
            "data": {
                "op": "i",
                "ts_ms": 999,
                "seq": 1,
                "payload": {
                    "source": {
                        "queryId": "searches-query",
                        "ts_ms": 999,
                    },
                    "after": {
                        "searchText": "incomprehensible search text",
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

    agent = _make_agent(pubsub_name=pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[pubsub_name], event_stream=events)

    drasi_trigger(agent, topic=drasi_topic)
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    cloudevent_ids = [
        c.kwargs["payload"]["_message_metadata"]["id"]
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == ["1", "2"]

    tasks = [
        json.loads(c.kwargs["payload"]["task"])
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    event_data = [e["data"] for e in events]
    assert tasks == event_data


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_ignores_malformed_events():
    """Test that the Drasi extension ignores events that do not conform to the expected format."""
    pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_topic = "differenttopic"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": pubsub_name,
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
            "pubsubname": pubsub_name,
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

    agent = _make_agent(pubsub_name=pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[pubsub_name], event_stream=events)

    drasi_trigger(agent, topic=drasi_topic)
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_filters_by_single_query_id():
    """Test that the Drasi extension correctly filters events for a single Drasi query ID."""
    pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_topic = "differenttopic"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "query1",
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
            "pubsubname": pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "query2",
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
            "id": "2",
            "specversion": "1.0",
            "datacontenttype": "application/json; charset=utf-8",
            "source": "test-publisher",
            "type": "com.dapr.event.sent",
        },
    ]

    agent = _make_agent(pubsub_name=pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[pubsub_name], event_stream=events)

    drasi_trigger(agent, topic=drasi_topic, query_id="query1")
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 1  # type: ignore[attr-defined]

    cloudevent_ids = [
        c.kwargs["payload"]["_message_metadata"]["id"]
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == ["1"]


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_filters_by_single_operation():
    """Test that the Drasi extension correctly filters events for a single Drasi operation."""
    pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_topic = "differenttopic"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": pubsub_name,
            "data": {
                "op": "i",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "query1",
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
            "pubsubname": pubsub_name,
            "data": {
                "op": "d",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "query2",
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

    agent = _make_agent(pubsub_name=pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[pubsub_name], event_stream=events)

    drasi_trigger(agent, topic=drasi_topic, operation="d")
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 1  # type: ignore[attr-defined]

    cloudevent_ids = [
        c.kwargs["payload"]["_message_metadata"]["id"]
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == ["2"]


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_filters_by_multiple_operations():
    """Test that the Drasi extension correctly filters events for multiple Drasi operations."""
    pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_topic = "differenttopic"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": pubsub_name,
            "data": {
                "op": "i",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "query1",
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
            "pubsubname": pubsub_name,
            "data": {
                "op": "d",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "query2",
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
            "pubsubname": pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "query3",
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

    agent = _make_agent(pubsub_name=pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[pubsub_name], event_stream=events)

    drasi_trigger(agent, topic=drasi_topic, operation=["i", "u"])
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 2  # type: ignore[attr-defined]

    cloudevent_ids = [
        c.kwargs["payload"]["_message_metadata"]["id"]
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == ["1", "3"]


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_filters_by_event_model():
    """Test that the Drasi extension correctly filters events by Drasi change event payloads."""

    class Counter(BaseModel):
        count: int

    pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_topic = "differenttopic"
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "query1",
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
            "pubsubname": pubsub_name,
            "data": {
                "op": "d",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "query1",
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
            "pubsubname": pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "query2",
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
            "pubsubname": pubsub_name,
            "data": {
                "op": "i",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "query1",
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

    agent = _make_agent(pubsub_name=pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[pubsub_name], event_stream=events)

    drasi_trigger(agent, topic=drasi_topic, event_model=Counter)
    runner.subscribe(agent)

    scheduler_method = runner.run_sync
    assert scheduler_method.call_count == 3  # type: ignore[attr-defined]

    cloudevent_ids = [
        c.kwargs["payload"]["_message_metadata"]["id"]
        for c in scheduler_method.call_args_list  # type: ignore[attr-defined]
    ]
    assert cloudevent_ids == ["1", "2", "4"]


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_raises_when_given_pubsub_is_not_registered():
    """Test that the Drasi extension fails when the given pub/sub component is not registered."""
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
                        "queryId": "test-query",
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

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    drasi_trigger(agent, pubsub=drasi_pubsub_name, topic=drasi_topic)

    # Ensure that the activation doesn't try to fall back to the agent's pub/sub component
    with pytest.raises(RuntimeError):
        runner.subscribe(agent)


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_raises_when_pubsub_matches_agent_pubsub():
    """Test that the Drasi extension fails when the agent pub/sub component is used and pub/sub topic matches the agent's pub/sub topic."""
    pubsub_name = "testpubsub"
    agent_topic = "testtopic"
    drasi_topic = agent_topic
    events = [
        {
            "topic": drasi_topic,
            "pubsubname": pubsub_name,
            "data": {
                "op": "u",
                "ts_ms": 0,
                "seq": 0,
                "payload": {
                    "source": {
                        "queryId": "test-query",
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

    agent = _make_agent(pubsub_name=pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[pubsub_name], event_stream=events)

    drasi_trigger(agent, topic=drasi_topic)

    with pytest.raises(RuntimeError):
        runner.subscribe(agent)


@pytest.mark.asyncio
@pytest.mark.ext
async def test_drasi_trigger_raises_when_given_pubsub_matches_agent_pubsub():
    """Test that the Drasi extension fails when the given pub/sub (component, topic) matches the agent's pub/sub (component, topic)."""
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
                        "queryId": "test-query",
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

    agent = _make_agent(pubsub_name=agent_pubsub_name, topic=agent_topic)
    runner = _make_runner(pubsub_names=[agent_pubsub_name], event_stream=events)

    drasi_trigger(agent, pubsub=drasi_pubsub_name, topic=drasi_topic)

    with pytest.raises(RuntimeError):
        runner.subscribe(agent)
