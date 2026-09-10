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

"""Regression tests for AgentRunner pub/sub re-wiring (issue #707).

A later call to `_wire_pubsub_routes` with a broader pubsub config (e.g. a
broadcast topic added after the first wiring) must register the new specs
instead of being skipped entirely by the instance-wide `_wired_pubsub` guard.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from dapr_agents.agents.configs import AgentPubSubConfig
from dapr_agents.workflow.decorators.decorators import message_router
from dapr_agents.workflow.runners.agent import AgentRunner

_WIRE_KWARGS = dict(
    delivery_mode="sync",
    queue_maxsize=10,
    await_result=False,
    await_timeout=None,
    fetch_payloads=True,
    log_outcome=False,
)


class _Message(BaseModel):
    text: str


class _FakeAgent:
    """Minimal stand-in for a DurableAgent with pub/sub handlers."""

    name = "fake-agent"

    @message_router(message_model=_Message)
    def handle_direct(self, message: _Message) -> None: ...

    @message_router(message_model=_Message, broadcast=True)
    def handle_broadcast(self, message: _Message) -> None: ...


def _make_runner() -> AgentRunner:
    with patch(
        "dapr_agents.workflow.runners.base.DaprWorkflowClient",
        return_value=MagicMock(),
    ):
        return AgentRunner(
            name="test-runner",
            wf_client=MagicMock(),
            client_factory=lambda: MagicMock(),
        )


def test_second_wire_call_registers_newly_added_broadcast_topic():
    agent = _FakeAgent()
    agent.pubsub = AgentPubSubConfig(pubsub_name="pubsub", agent_topic="direct-topic")
    runner = _make_runner()

    with patch(
        "dapr_agents.workflow.runners.agent.register_message_routes",
        return_value=[],
    ) as mock_register:
        runner._wire_pubsub_routes(agent=agent, **_WIRE_KWARGS)
        assert mock_register.call_count == 1
        first_topics = {s.topic for s in mock_register.call_args.kwargs["routes"]}
        assert first_topics == {"direct-topic"}

        # Broadcast topic configured after the agent was already wired once.
        agent.pubsub = AgentPubSubConfig(
            pubsub_name="pubsub",
            agent_topic="direct-topic",
            broadcast_topic="broadcast-topic",
        )
        runner._wire_pubsub_routes(agent=agent, **_WIRE_KWARGS)

    assert mock_register.call_count == 2
    second_topics = {s.topic for s in mock_register.call_args.kwargs["routes"]}
    assert second_topics == {"broadcast-topic"}


def test_second_wire_call_with_no_new_topics_is_a_noop():
    agent = _FakeAgent()
    agent.pubsub = AgentPubSubConfig(pubsub_name="pubsub", agent_topic="direct-topic")
    runner = _make_runner()

    with patch(
        "dapr_agents.workflow.runners.agent.register_message_routes",
        return_value=[],
    ) as mock_register:
        runner._wire_pubsub_routes(agent=agent, **_WIRE_KWARGS)
        runner._wire_pubsub_routes(agent=agent, **_WIRE_KWARGS)

    assert mock_register.call_count == 1


def test_unwire_pubsub_clears_wired_topics():
    agent = _FakeAgent()
    agent.pubsub = AgentPubSubConfig(pubsub_name="pubsub", agent_topic="direct-topic")
    runner = _make_runner()

    with patch(
        "dapr_agents.workflow.runners.agent.register_message_routes",
        return_value=[],
    ) as mock_register:
        runner._wire_pubsub_routes(agent=agent, **_WIRE_KWARGS)
        runner.unwire_pubsub()
        runner._wire_pubsub_routes(agent=agent, **_WIRE_KWARGS)

    assert mock_register.call_count == 2
