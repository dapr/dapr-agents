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

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dapr_agents.workflow.utils.core import call_agent, trigger_agent


# ---------------------------------------------------------------------------
# call_agent tests
# ---------------------------------------------------------------------------


def _make_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.call_child_workflow.return_value = MagicMock()
    return ctx


def test_call_agent_constructs_correct_workflow_name():
    ctx = _make_ctx()
    call_agent(ctx, "WeatherAgent", input={"task": "test"}, app_id="weather-agent")
    kwargs = ctx.call_child_workflow.call_args.kwargs
    assert kwargs["workflow"] == "dapr.agents.WeatherAgent.workflow"


def test_call_agent_sanitizes_name():
    ctx = _make_ctx()
    call_agent(ctx, "weather agent", input={}, app_id="weather-agent")
    kwargs = ctx.call_child_workflow.call_args.kwargs
    assert kwargs["workflow"] == "dapr.agents.WeatherAgent.workflow"


def test_call_agent_passes_app_id():
    ctx = _make_ctx()
    call_agent(ctx, "WeatherAgent", input={}, app_id="my-app")
    kwargs = ctx.call_child_workflow.call_args.kwargs
    assert kwargs["app_id"] == "my-app"


def test_call_agent_passes_input():
    ctx = _make_ctx()
    payload = {"task": "What is the weather?"}
    call_agent(ctx, "WeatherAgent", input=payload, app_id="weather-agent")
    kwargs = ctx.call_child_workflow.call_args.kwargs
    assert kwargs["input"] == payload


def test_call_agent_with_instance_id():
    ctx = _make_ctx()
    call_agent(
        ctx, "WeatherAgent", input={}, app_id="weather-agent", instance_id="abc-123"
    )
    kwargs = ctx.call_child_workflow.call_args.kwargs
    assert kwargs["instance_id"] == "abc-123"


def test_call_agent_without_instance_id():
    ctx = _make_ctx()
    call_agent(ctx, "WeatherAgent", input={}, app_id="weather-agent")
    kwargs = ctx.call_child_workflow.call_args.kwargs
    assert "instance_id" not in kwargs


def test_call_agent_with_framework():
    ctx = _make_ctx()
    call_agent(
        ctx, "WeatherAgent", input={}, app_id="weather-agent", framework="openai"
    )
    kwargs = ctx.call_child_workflow.call_args.kwargs
    assert kwargs["workflow"] == "dapr.openai.WeatherAgent.workflow"


def test_call_agent_returns_task():
    ctx = _make_ctx()
    task = MagicMock()
    ctx.call_child_workflow.return_value = task
    result = call_agent(ctx, "WeatherAgent", input={}, app_id="weather-agent")
    assert result is task


def test_call_agent_raises_without_app_id_and_no_registry():
    ctx = _make_ctx()
    with pytest.raises(ValueError, match="app_id is required"):
        call_agent(ctx, "WeatherAgent", input={})


def test_call_agent_looks_up_app_id_from_registry():
    ctx = _make_ctx()
    registry = MagicMock()
    registry.get_agents_metadata.return_value = {
        "WeatherAgent": {"agent": {"appid": "weather-agent", "framework": None}}
    }
    call_agent(ctx, "WeatherAgent", input={}, registry=registry)
    kwargs = ctx.call_child_workflow.call_args.kwargs
    assert kwargs["app_id"] == "weather-agent"


def test_call_agent_looks_up_framework_from_registry():
    ctx = _make_ctx()
    registry = MagicMock()
    registry.get_agents_metadata.return_value = {
        "WeatherAgent": {"agent": {"appid": "weather-agent", "framework": "openai"}}
    }
    call_agent(ctx, "WeatherAgent", input={}, registry=registry)
    kwargs = ctx.call_child_workflow.call_args.kwargs
    assert kwargs["workflow"] == "dapr.openai.WeatherAgent.workflow"


def test_call_agent_explicit_app_id_overrides_registry():
    ctx = _make_ctx()
    registry = MagicMock()
    registry.get_agents_metadata.return_value = {
        "WeatherAgent": {"agent": {"appid": "registry-app", "framework": None}}
    }
    call_agent(ctx, "WeatherAgent", input={}, app_id="explicit-app", registry=registry)
    kwargs = ctx.call_child_workflow.call_args.kwargs
    assert kwargs["app_id"] == "explicit-app"


# ---------------------------------------------------------------------------
# trigger_agent tests
# ---------------------------------------------------------------------------


def _make_wf_mocks():
    """Return (mock_wfr, mock_client, mock_state) and patch dapr.ext.workflow."""
    mock_state = MagicMock()
    mock_state.runtime_status.name = "COMPLETED"
    mock_state.serialized_output = '{"answer": "Sunny"}'

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.schedule_new_workflow.return_value = "instance-123"
    mock_client.wait_for_workflow_completion.return_value = mock_state

    mock_wfr = MagicMock()

    return mock_wfr, mock_client, mock_state


def test_trigger_agent_registers_and_starts_runtime():
    mock_wfr, mock_client, mock_state = _make_wf_mocks()

    with (
        patch("dapr.ext.workflow.WorkflowRuntime", return_value=mock_wfr),
        patch("dapr.ext.workflow.DaprWorkflowClient", return_value=mock_client),
    ):
        trigger_agent("WeatherAgent", input={}, app_id="weather-agent")

    mock_wfr.register_workflow.assert_called_once()
    mock_wfr.start.assert_called_once()


def test_trigger_agent_schedules_workflow():
    mock_wfr, mock_client, mock_state = _make_wf_mocks()

    with (
        patch("dapr.ext.workflow.WorkflowRuntime", return_value=mock_wfr),
        patch("dapr.ext.workflow.DaprWorkflowClient", return_value=mock_client),
    ):
        trigger_agent("WeatherAgent", input={}, app_id="weather-agent")

    mock_client.schedule_new_workflow.assert_called_once()


def test_trigger_agent_waits_for_completion():
    mock_wfr, mock_client, mock_state = _make_wf_mocks()

    with (
        patch("dapr.ext.workflow.WorkflowRuntime", return_value=mock_wfr),
        patch("dapr.ext.workflow.DaprWorkflowClient", return_value=mock_client),
    ):
        trigger_agent(
            "WeatherAgent", input={}, app_id="weather-agent", timeout_in_seconds=60
        )

    mock_client.wait_for_workflow_completion.assert_called_once_with(
        instance_id="instance-123",
        timeout_in_seconds=60,
        fetch_payloads=True,
    )


def test_trigger_agent_shuts_down_on_success():
    mock_wfr, mock_client, mock_state = _make_wf_mocks()

    with (
        patch("dapr.ext.workflow.WorkflowRuntime", return_value=mock_wfr),
        patch("dapr.ext.workflow.DaprWorkflowClient", return_value=mock_client),
    ):
        trigger_agent("WeatherAgent", input={}, app_id="weather-agent")

    mock_wfr.shutdown.assert_called_once()


def test_trigger_agent_shuts_down_on_exception():
    mock_wfr, mock_client, _ = _make_wf_mocks()
    mock_client.wait_for_workflow_completion.side_effect = RuntimeError("timeout")

    with (
        patch("dapr.ext.workflow.WorkflowRuntime", return_value=mock_wfr),
        patch("dapr.ext.workflow.DaprWorkflowClient", return_value=mock_client),
    ):
        with pytest.raises(RuntimeError):
            trigger_agent("WeatherAgent", input={}, app_id="weather-agent")

    # shutdown must still be called even when an exception is raised
    mock_wfr.shutdown.assert_called_once()


def test_trigger_agent_timeout_logs_warning_and_returns_none():
    mock_wfr, mock_client, _ = _make_wf_mocks()
    mock_client.wait_for_workflow_completion.side_effect = TimeoutError("timed out")

    with (
        patch("dapr.ext.workflow.WorkflowRuntime", return_value=mock_wfr),
        patch("dapr.ext.workflow.DaprWorkflowClient", return_value=mock_client),
        patch("dapr_agents.workflow.utils.core.logger") as mock_logger,
    ):
        result = trigger_agent(
            "WeatherAgent", input={}, app_id="weather-agent", timeout_in_seconds=5
        )

    assert result is None
    mock_logger.warning.assert_called_once()
    mock_wfr.shutdown.assert_called_once()


def test_trigger_agent_returns_serialized_output():
    mock_wfr, mock_client, mock_state = _make_wf_mocks()
    mock_state.serialized_output = '{"answer": "Sunny"}'

    with (
        patch("dapr.ext.workflow.WorkflowRuntime", return_value=mock_wfr),
        patch("dapr.ext.workflow.DaprWorkflowClient", return_value=mock_client),
    ):
        result = trigger_agent("WeatherAgent", input={}, app_id="weather-agent")

    assert result == '{"answer": "Sunny"}'


def test_trigger_agent_raises_without_app_id_and_no_registry():
    with pytest.raises(ValueError, match="app_id is required"):
        trigger_agent("WeatherAgent", input={})


def test_trigger_agent_looks_up_app_id_from_registry():
    mock_wfr, mock_client, _ = _make_wf_mocks()
    registry = MagicMock()
    registry.get_agents_metadata.return_value = {
        "WeatherAgent": {"agent": {"appid": "weather-agent", "framework": None}}
    }

    with (
        patch("dapr.ext.workflow.WorkflowRuntime", return_value=mock_wfr),
        patch("dapr.ext.workflow.DaprWorkflowClient", return_value=mock_client),
    ):
        trigger_agent("WeatherAgent", input={}, registry=registry)

    mock_client.schedule_new_workflow.assert_called_once()


def test_trigger_agent_wrapper_workflow_name():
    """Trigger wrapper follows dapr.agents.<name>.trigger convention."""
    mock_wfr, mock_client, _ = _make_wf_mocks()
    registered_fn = None

    def capture_register(fn):
        nonlocal registered_fn
        registered_fn = fn

    mock_wfr.register_workflow.side_effect = capture_register

    with (
        patch("dapr.ext.workflow.WorkflowRuntime", return_value=mock_wfr),
        patch("dapr.ext.workflow.DaprWorkflowClient", return_value=mock_client),
    ):
        trigger_agent("WeatherAgent", input={}, app_id="weather-agent")

    assert registered_fn is not None
    assert registered_fn.__name__ == "dapr.agents.WeatherAgent.trigger"


# ---------------------------------------------------------------------------
# Root import test
# ---------------------------------------------------------------------------


def test_importable_from_root():
    from dapr_agents import call_agent as ca, trigger_agent as ta  # noqa: F401

    assert callable(ca)
    assert callable(ta)
