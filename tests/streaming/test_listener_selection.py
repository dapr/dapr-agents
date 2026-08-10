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

"""Default stream-listener selection in ``AgentRunner._resolve_default_listener``.

Precedence: explicit > ``execution.stream_listener`` > pubsub-when-bus-configured
> in_process. The bus-configured default makes deployed agents safe across a
multi-replica handoff without an env var, while local/no-broker runs still use
the in-process queue.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from dapr_agents.workflow.runners.agent import AgentRunner

_MULTI_REPLICA_ENV = (
    "DAPR_AGENTS_MULTI_REPLICA",
    "DAPR_AGENTS_REPLICA_COUNT",
    "REPLICA_COUNT",
)


def _clear_replica_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _MULTI_REPLICA_ENV:
        monkeypatch.delenv(var, raising=False)


class _FakeInfra:
    def __init__(self, message_bus_name: Optional[str]) -> None:
        self.message_bus_name = message_bus_name

    def stream_topic_name(self, root_instance_id: str) -> str:
        return f"stream-{root_instance_id}"


def _make_agent(
    *,
    message_bus_name: Optional[str],
    stream_listener: Any = None,
    name: str = "sel-agent",
) -> SimpleNamespace:
    # tool_executor=None => _agent_has_cross_app_peers() returns False.
    return SimpleNamespace(
        name=name,
        execution=SimpleNamespace(stream_listener=stream_listener),
        _infra=_FakeInfra(message_bus_name),
        tool_executor=None,
    )


@pytest.fixture
def runner() -> AgentRunner:
    # Bypass __init__ — the selection helpers only touch the passed-in agent.
    return AgentRunner.__new__(AgentRunner)


def test_default_prefers_pubsub_when_bus_configured(runner, monkeypatch) -> None:
    _clear_replica_env(monkeypatch)
    agent = _make_agent(message_bus_name="messagepubsub", name="bus-agent")
    cfg = runner._resolve_default_listener(
        agent=agent, explicit=None, root_instance_id="root-1"
    )
    assert cfg["type"] == "pubsub"
    assert cfg["pubsub_name"] == "messagepubsub"
    assert cfg["topic"] == "stream-root-1"


def test_default_falls_back_to_in_process_without_bus(
    runner, monkeypatch, caplog
) -> None:
    _clear_replica_env(monkeypatch)
    from dapr_agents.workflow.runners.agent import _WARNED_IN_PROCESS_DEFAULT

    _WARNED_IN_PROCESS_DEFAULT.discard("nobus-agent")
    agent = _make_agent(message_bus_name=None, name="nobus-agent")
    with caplog.at_level(logging.WARNING):
        cfg = runner._resolve_default_listener(
            agent=agent, explicit=None, root_instance_id="root-2"
        )
    assert cfg["type"] == "in_process"
    assert cfg["registry_key"] == "root-2"
    # Warns once so scaled deployments know in_process is unsafe across handoff.
    assert "in_process" in caplog.text
    assert "nobus-agent" in caplog.text


def test_explicit_config_wins_over_bus_default(runner, monkeypatch) -> None:
    _clear_replica_env(monkeypatch)
    agent = _make_agent(message_bus_name="messagepubsub", name="explicit-agent")
    cfg = runner._resolve_default_listener(
        agent=agent, explicit={"type": "in_process"}, root_instance_id="root-3"
    )
    assert cfg["type"] == "in_process"


def test_execution_stream_listener_wins_over_bus_default(runner, monkeypatch) -> None:
    _clear_replica_env(monkeypatch)
    agent = _make_agent(
        message_bus_name="messagepubsub",
        stream_listener={"type": "in_process"},
        name="configured-agent",
    )
    cfg = runner._resolve_default_listener(
        agent=agent, explicit=None, root_instance_id="root-4"
    )
    assert cfg["type"] == "in_process"


def test_multi_replica_env_forces_pubsub(runner, monkeypatch) -> None:
    _clear_replica_env(monkeypatch)
    monkeypatch.setenv("DAPR_AGENTS_MULTI_REPLICA", "1")
    agent = _make_agent(message_bus_name="messagepubsub", name="mr-agent")
    cfg = runner._resolve_default_listener(
        agent=agent, explicit=None, root_instance_id="root-5"
    )
    assert cfg["type"] == "pubsub"


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, False),
        ({"DAPR_AGENTS_MULTI_REPLICA": "1"}, True),
        ({"DAPR_AGENTS_MULTI_REPLICA": "true"}, True),
        ({"DAPR_AGENTS_MULTI_REPLICA": "0"}, False),
        ({"DAPR_AGENTS_REPLICA_COUNT": "3"}, True),
        ({"DAPR_AGENTS_REPLICA_COUNT": "1"}, False),
        ({"REPLICA_COUNT": "2"}, True),
    ],
)
def test_is_multi_replica_env_matrix(monkeypatch, env, expected) -> None:
    _clear_replica_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert AgentRunner._is_multi_replica() is expected
