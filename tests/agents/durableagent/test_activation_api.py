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

"""Unit tests for the DurableAgent.add_activation registration API."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Optional
from unittest.mock import MagicMock, Mock

import pytest

from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService


@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
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
    yield


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.base.default_dapr_client_factory",
        lambda: MagicMock(),
    )
    yield
    os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture
def agent() -> DurableAgent:
    llm = Mock(spec=OpenAIChatClient)
    llm.prompt_template = None
    llm.__class__.__name__ = "MockLLMClient"
    llm.provider = "MockOpenAIProvider"
    llm.api = "MockOpenAIAPI"
    llm.model = "gpt-4o-mock"
    return DurableAgent(
        name="ApiAgent",
        role="Test Assistant",
        goal="Help with testing",
        llm=llm,
        pubsub=AgentPubSubConfig(pubsub_name="testpubsub", agent_topic="ApiAgent"),
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
        registry=AgentRegistryConfig(
            store=StateStoreService(store_name="testregistry")
        ),
    )


def test_fresh_agent_has_no_activations(agent):
    assert agent.activations == []


def test_add_activation_stores_in_registration_order(agent):
    def a(ctx):
        return None

    def b(ctx):
        return None

    agent.add_activation(a)
    agent.add_activation(b)

    assert agent.activations == [a, b]


def test_activations_property_returns_a_copy(agent):
    agent.add_activation(lambda ctx: None)
    snapshot = agent.activations
    snapshot.append("intruder")

    assert len(agent.activations) == 1  # internal list untouched


def test_add_activation_rejects_non_callable(agent):
    with pytest.raises(TypeError):
        agent.add_activation("not-callable")


def test_add_activation_after_hosting_window_closed_raises(agent):
    # The runner closes this window on first attach; simulate that here.
    agent._activation_window_open = False

    with pytest.raises(RuntimeError):
        agent.add_activation(lambda ctx: None)
