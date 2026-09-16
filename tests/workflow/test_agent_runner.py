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

"""Regression tests for AgentRunner (issue #745).

#699 fixed a bug where hosting an agent would fail if its broadcast topic
was not explicitly provided, even though the broadcast topic is opt-in.
These tests make sure each AgentRunner entrypoint keeps working for an
agent with no broadcast topic configured.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import FastAPI

from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners.agent import AgentRunner


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.base.default_dapr_client_factory",
        lambda: MagicMock(),
    )
    yield
    os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture(autouse=True)
def patch_dapr_workflow_runtime(monkeypatch):
    """Mock the workflow runtime so no live Dapr instance is required."""
    import dapr.ext.workflow as wf

    monkeypatch.setattr(wf, "WorkflowRuntime", lambda: Mock(spec=wf.WorkflowRuntime))


@pytest.fixture(autouse=True)
def patch_agent_bootstrap_dapr_client(monkeypatch):
    monkeypatch.setattr(
        "dapr_agents.agents.base.DaprClient",
        lambda *args, **kwargs: MagicMock(),
    )


class _MockAgentMetadata:
    def __init__(self, *args, **kwargs):
        pass


class _MockAgentMetadataSchema:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def patch_agent_bootstrap_metadata(monkeypatch):
    monkeypatch.setattr("dapr_agents.agents.base.AgentMetadata", _MockAgentMetadata)
    monkeypatch.setattr(
        "dapr_agents.agents.base.AgentMetadataSchema", _MockAgentMetadataSchema
    )


@pytest.fixture(autouse=True)
def stub_agent_lifecycle(monkeypatch):
    monkeypatch.setattr(DurableAgent, "start", Mock())
    monkeypatch.setattr(DurableAgent, "stop", Mock())


@pytest.fixture(autouse=True)
def stub_route_wiring(monkeypatch):
    """Stub the actual subscription/HTTP mechanics; these tests only assert
    that the runner entrypoints complete without raising."""
    monkeypatch.setattr(
        "dapr_agents.workflow.runners.agent.register_message_routes",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        "dapr_agents.workflow.runners.agent.register_http_routes",
        Mock(return_value=[]),
    )


def _make_agent_without_broadcast_topic(name: str) -> DurableAgent:
    llm = Mock(spec=OpenAIChatClient)
    llm.prompt_template = None
    llm.__class__.__name__ = "MockLLMClient"
    llm.provider = "MockOpenAIProvider"
    llm.api = "MockOpenAIAPI"
    llm.model = "gpt-4o-mock"

    return DurableAgent(
        name=name,
        role="Test Assistant",
        goal="Help with testing",
        llm=llm,
        pubsub=AgentPubSubConfig(pubsub_name="testpubsub", agent_topic="testtopic"),
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
        execution=AgentExecutionConfig(max_iterations=5),
    )


def _make_runner() -> AgentRunner:
    runner = AgentRunner(
        wf_client=MagicMock(),
        client_factory=Mock(side_effect=lambda: MagicMock()),
    )
    runner._wire_http_routes = Mock()
    runner._mount_service_routes = Mock()
    runner._mount_hitl_routes = Mock()
    return runner


def test_register_routes_without_broadcast_topic():
    agent = _make_agent_without_broadcast_topic("RegisterRoutesAgent")
    runner = _make_runner()
    try:
        runner.register_routes(agent, fastapi_app=FastAPI())
    finally:
        runner.shutdown(agent)
        agent.stop()


def test_subscribe_without_broadcast_topic():
    agent = _make_agent_without_broadcast_topic("SubscribeAgent")
    runner = _make_runner()
    try:
        result = runner.subscribe(agent)
        assert result is runner
    finally:
        runner.shutdown(agent)
        agent.stop()


def test_serve_without_broadcast_topic():
    agent = _make_agent_without_broadcast_topic("ServeAgent")
    runner = _make_runner()
    try:
        app = runner.serve(agent, app=FastAPI())
        assert app is not None
    finally:
        runner.shutdown(agent)
        agent.stop()
