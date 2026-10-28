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

"""The agents' workflows are registered as named versions.

See "Changing workflow code" in docs/development/README.md.
"""

from typing import Any, Dict, Tuple
from unittest.mock import MagicMock

from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
    OrchestrationMode,
)
from dapr_agents.agents.durable import DurableAgent, orchestration_workflow_id
from dapr_agents.storage.daprstores.stateservice import StateStoreService


def _agent(broadcast_topic=None, execution=None) -> DurableAgent:
    return DurableAgent(
        name="Versioned",
        role="Tester",
        goal="Test",
        instructions=["Be brief"],
        pubsub=AgentPubSubConfig(pubsub_name="p", broadcast_topic=broadcast_topic),
        state=AgentStateConfig(store=StateStoreService(store_name="s")),
        execution=execution,
    )


def _registered(agent: DurableAgent) -> Dict[str, Tuple[str, bool, Any]]:
    runtime = MagicMock()
    agent.register_workflows(runtime)
    assert runtime.register_workflow.call_count == 0
    return {
        c.kwargs["name"]: (c.kwargs["version_name"], c.kwargs["is_latest"], c.args[0])
        for c in runtime.register_versioned_workflow.call_args_list
    }


def test_agent_workflow_is_registered_as_latest_v1():
    agent = _agent()
    registered = _registered(agent)

    assert set(registered) == {agent.agent_workflow_name}
    version, latest, fn = registered[agent.agent_workflow_name]
    assert (version, latest) == ("v1", True)
    assert fn.__name__ == agent.agent_workflow_name
    assert fn.__wrapped__ == agent.agent_workflow


def test_broadcast_and_orchestration_are_versioned_when_configured():
    agent = _agent(
        broadcast_topic="team",
        execution=AgentExecutionConfig(orchestration_mode=OrchestrationMode.AGENT),
    )
    registered = _registered(agent)

    assert set(registered) == {
        agent.agent_workflow_name,
        agent.broadcast_workflow_name,
        orchestration_workflow_id(agent.name, infra=agent._infra),
    }
    assert {v[:2] for v in registered.values()} == {("v1", True)}


def test_only_the_last_version_is_latest():
    agent = _agent()
    runtime = MagicMock()

    agent._register_versions(
        runtime, "wf", (("v1", agent.agent_workflow), ("v2", agent.on_broadcast))
    )

    assert [
        (c.kwargs["name"], c.kwargs["version_name"], c.kwargs["is_latest"])
        for c in runtime.register_versioned_workflow.call_args_list
    ] == [("wf", "v1", False), ("wf", "v2", True)]
