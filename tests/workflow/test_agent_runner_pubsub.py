from unittest.mock import MagicMock, patch

from dapr_agents.types.workflow import PubSubRouteSpec
from dapr_agents.workflow.runners.agent import AgentRunner


def _wire(runner, agent):
    runner._wire_pubsub_routes(
        agent=agent,
        delivery_mode="sync",
        queue_maxsize=1024,
        await_result=False,
        await_timeout=None,
        fetch_payloads=True,
        log_outcome=False,
    )


def test_rewires_pubsub_routes_when_specs_change():
    runner = AgentRunner(wf_client=MagicMock(), client_factory=lambda: MagicMock())
    agent = MagicMock(pubsub=object())
    first_handler = MagicMock()
    second_handler = MagicMock()
    first = [PubSubRouteSpec("pubsub", "first", first_handler)]
    second = [PubSubRouteSpec("pubsub", "second", second_handler)]
    runner._build_pubsub_specs = MagicMock(side_effect=[first, second])
    first_close = MagicMock()
    second_close = MagicMock()

    with patch(
        "dapr_agents.workflow.runners.agent.register_message_routes",
        side_effect=[[first_close], [second_close]],
    ) as register:
        _wire(runner, agent)
        _wire(runner, agent)

    first_close.assert_called_once_with()
    assert register.call_count == 2
    assert runner._pubsub_specs == second
    assert runner._pubsub_closers == [second_close]


def test_does_not_reregister_identical_pubsub_specs():
    runner = AgentRunner(wf_client=MagicMock(), client_factory=lambda: MagicMock())
    agent = MagicMock(pubsub=object())
    handler = MagicMock()
    specs = [PubSubRouteSpec("pubsub", "topic", handler)]
    runner._build_pubsub_specs = MagicMock(side_effect=[specs, specs.copy()])
    close = MagicMock()

    with patch(
        "dapr_agents.workflow.runners.agent.register_message_routes",
        return_value=[close],
    ) as register:
        _wire(runner, agent)
        _wire(runner, agent)

    register.assert_called_once()
    close.assert_not_called()
