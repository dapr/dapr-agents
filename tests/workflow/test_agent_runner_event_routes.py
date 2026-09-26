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

"""AgentRunner / WorkflowRunner wiring for workflow event routes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from dapr_agents.agents.configs import AgentPubSubConfig
from dapr_agents.types.workflow import HttpRouteSpec, WorkflowEventRouteSpec
from dapr_agents.workflow.decorators.decorators import message_router
from dapr_agents.workflow.runners.agent import AgentRunner
from dapr_agents.workflow.runners.base import WorkflowRunner

_REGISTER = "dapr_agents.workflow.runners.agent.register_message_routes"

_WIRE_KWARGS: dict[str, Any] = dict(
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
    name = "fake-agent"

    @message_router(message_model=_Message)
    def handle_direct(self, message: _Message) -> None: ...


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


def _agent(pubsub: bool = True) -> _FakeAgent:
    agent = _FakeAgent()
    agent.pubsub = (  # type: ignore[attr-defined]
        AgentPubSubConfig(pubsub_name="pubsub", agent_topic="direct-topic")
        if pubsub
        else None
    )
    return agent


def _event_spec(topic: str = "jobs", event_name: str = "job_finished"):
    return WorkflowEventRouteSpec(
        pubsub_name="pubsub",
        topic=topic,
        event_name=event_name,
        instance_id_from="workflow_id",
    )


def _routes(mock_register: MagicMock) -> list[Any]:
    return list(mock_register.call_args.kwargs["routes"])


def test_wire_registers_agent_and_event_specs_together():
    runner, spec = _make_runner(), _event_spec()
    with patch(_REGISTER, return_value=[]) as mock_register:
        runner._wire_pubsub_routes(agent=_agent(), event_routes=[spec], **_WIRE_KWARGS)
    assert mock_register.call_count == 1
    routes = _routes(mock_register)
    assert {getattr(r, "topic") for r in routes} == {"direct-topic", "jobs"}
    assert spec in routes


def test_agent_without_pubsub_still_registers_event_routes():
    runner, spec = _make_runner(), _event_spec()
    with patch(_REGISTER, return_value=[]) as mock_register:
        runner._wire_pubsub_routes(
            agent=_agent(pubsub=False), event_routes=[spec], **_WIRE_KWARGS
        )
    assert _routes(mock_register) == [spec]


def test_rewiring_same_event_route_is_noop():
    runner, agent, spec = _make_runner(), _agent(), _event_spec()
    with patch(_REGISTER, return_value=[]) as mock_register:
        runner._wire_pubsub_routes(agent=agent, event_routes=[spec], **_WIRE_KWARGS)
        runner._wire_pubsub_routes(agent=agent, event_routes=[spec], **_WIRE_KWARGS)
    assert mock_register.call_count == 1


def test_later_event_route_is_added():
    runner, agent = _make_runner(), _agent()
    with patch(_REGISTER, return_value=[]) as mock_register:
        runner._wire_pubsub_routes(agent=agent, **_WIRE_KWARGS)
        runner._wire_pubsub_routes(
            agent=agent, event_routes=[_event_spec()], **_WIRE_KWARGS
        )
    assert mock_register.call_count == 2
    assert [r.topic for r in _routes(mock_register)] == ["jobs"]


def test_same_topic_different_event_name_rejected():
    runner, agent = _make_runner(), _agent()
    with patch(_REGISTER, return_value=[]):
        runner._wire_pubsub_routes(
            agent=agent, event_routes=[_event_spec()], **_WIRE_KWARGS
        )
        with pytest.raises(ValueError, match="already carries"):
            runner._wire_pubsub_routes(
                agent=agent,
                event_routes=[_event_spec(event_name="other")],
                **_WIRE_KWARGS,
            )


def test_event_route_on_agent_topic_rejected_in_same_call():
    runner = _make_runner()
    with patch(_REGISTER, return_value=[]) as mock_register:
        with pytest.raises(ValueError, match="agent pub/sub topic"):
            runner._wire_pubsub_routes(
                agent=_agent(),
                event_routes=[_event_spec(topic="direct-topic")],
                **_WIRE_KWARGS,
            )
    mock_register.assert_not_called()


def test_event_route_on_agent_topic_rejected_after_wiring():
    runner, agent = _make_runner(), _agent()
    with patch(_REGISTER, return_value=[]):
        runner._wire_pubsub_routes(agent=agent, **_WIRE_KWARGS)
        with pytest.raises(ValueError, match="agent pub/sub topic"):
            runner._wire_pubsub_routes(
                agent=agent,
                event_routes=[_event_spec(topic="direct-topic")],
                **_WIRE_KWARGS,
            )


def test_agent_topic_on_wired_event_topic_rejected():
    runner = _make_runner()
    with patch(_REGISTER, return_value=[]):
        runner._wire_pubsub_routes(
            agent=_agent(pubsub=False),
            event_routes=[_event_spec(topic="direct-topic")],
            **_WIRE_KWARGS,
        )
        with pytest.raises(ValueError, match="already carries workflow event route"):
            runner._wire_pubsub_routes(agent=_agent(), **_WIRE_KWARGS)


def test_two_event_specs_same_topic_rejected():
    runner = _make_runner()
    with patch(_REGISTER, return_value=[]):
        with pytest.raises(ValueError, match="one topic per event route"):
            runner._wire_pubsub_routes(
                agent=_agent(),
                event_routes=[_event_spec(), _event_spec(event_name="b")],
                **_WIRE_KWARGS,
            )


def test_unwire_clears_event_routes():
    runner, agent, spec = _make_runner(), _agent(), _event_spec()
    with patch(_REGISTER, return_value=[]) as mock_register:
        runner._wire_pubsub_routes(agent=agent, event_routes=[spec], **_WIRE_KWARGS)
        runner.unwire_pubsub()
        assert runner._wired_event_routes == {}
        runner._wire_pubsub_routes(agent=agent, event_routes=[spec], **_WIRE_KWARGS)
    assert mock_register.call_count == 2


@pytest.mark.parametrize("entry", ["subscribe", "register_routes", "serve"])
def test_public_entry_points_forward_event_routes(entry):
    runner, agent, spec = _make_runner(), _agent(), _event_spec()
    with (
        patch.object(runner, "_wire_pubsub_routes") as mock_wire,
        patch.object(runner, "_attach_agent"),
        patch.object(runner, "_mount_hitl_routes"),
    ):
        if entry == "serve":
            runner.serve(agent, app=FastAPI(), event_routes=[spec], expose_entry=False)
        else:
            getattr(runner, entry)(agent, event_routes=[spec])
    assert mock_wire.call_args.kwargs["event_routes"] == [spec]


def test_workflow_runner_explicit_mode_splits_event_and_http_specs():
    with patch(
        "dapr_agents.workflow.runners.base.DaprWorkflowClient",
        return_value=MagicMock(),
    ):
        runner = WorkflowRunner(
            name="wr", wf_client=MagicMock(), client_factory=lambda: MagicMock()
        )
    runner._dapr_client = MagicMock()
    spec = _event_spec()

    def start(body: dict) -> dict:
        return body

    http_spec = HttpRouteSpec(path="/start", handler_fn=start)
    with (
        patch(
            "dapr_agents.workflow.runners.base.register_message_routes",
            return_value=[],
        ) as mock_msg,
        patch("dapr_agents.workflow.runners.base.register_http_routes") as mock_http,
    ):
        runner.register_routes(routes=[spec, http_spec], fastapi_app=FastAPI())
    assert mock_msg.call_args.kwargs["routes"] == [spec]
    assert mock_http.call_args.kwargs["routes"] == [http_spec]
