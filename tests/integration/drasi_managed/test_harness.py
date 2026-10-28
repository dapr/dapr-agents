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

"""Offline regressions for the harness, separate from real-runtime coverage."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from . import model_app
from .runtime import HERE, ROUTER_REVISION, Images, Runtime


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Runtime]:
    harness = Runtime(
        tmp_path,
        Images(agent="unused-agent", router="unused-router", core_version="test-only"),
    )
    try:
        yield harness
    finally:
        harness.client.close()


def test_harness_does_not_copy_host_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-inherited")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "must-not-be-inherited")
    monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION", "invented-version")
    monkeypatch.setenv("UV_SYSTEM_CERTS", "true")
    runtime = Runtime(
        tmp_path,
        Images(agent="unused-agent", router="unused-router", core_version="test-only"),
    )
    try:
        for name in (
            "OPENAI_API_KEY",
            "AZURE_CLIENT_SECRET",
            "SETUPTOOLS_SCM_PRETEND_VERSION",
        ):
            assert name not in runtime.environment
        assert runtime.environment["DRASI_CORE_VERSION"] == "test-only"
        assert runtime.environment["UV_SYSTEM_CERTS"] == "true"
        assert "must-not-be-inherited" not in (tmp_path / "compose.env").read_text()
    finally:
        runtime.client.close()


def test_compose_preserves_pins_private_dns_and_derived_version() -> None:
    compose = yaml.safe_load((HERE / "compose.yaml").read_text())
    services = compose["services"]
    router_build = services["router"]["build"]
    prefix = "https://github.com/drasi-project/drasi-platform.git#"
    assert router_build["context"] == (
        f"{prefix}{ROUTER_REVISION}:reactions/dapr/agent-router"
    )
    assert router_build["additional_contexts"]["reaction_sdk"] == (
        f"{prefix}{ROUTER_REVISION}:reactions/sdk/python"
    )
    assert "DRASI_CORE_VERSION" in services["model"]["build"]["args"]
    for service in services.values():
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1::")
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    assert config["spec"]["nameResolution"] == {
        "component": "nameformat",
        "version": "v1",
        "configuration": {"format": "{appid}-dapr:50002"},
    }
    for service, app_id in (
        ("router-dapr", "drasi-router"),
        ("agent-dapr", "managed-agent"),
    ):
        assert f"{app_id}-dapr" in services[service]["networks"]["default"]["aliases"]


@pytest.fixture
def start_calls(harness: Runtime, monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    calls = {
        "compose": Mock(return_value="container logs"),
        "url": Mock(return_value="http://127.0.0.1:12345"),
        "request": Mock(return_value={"inbox": "inbox", "dlt": "dlt"}),
        "record_versions": Mock(),
        "scheduling": Mock(return_value=[]),
        "evidence": Mock(return_value={"calls": [], "records": []}),
        "stream": Mock(return_value=[]),
    }
    for name, call in calls.items():
        monkeypatch.setattr(harness, name, call)
    return calls


@pytest.mark.parametrize("failure", ["ports", "readiness", "versions"])
def test_partial_startup_captures_only_logs(
    harness: Runtime, start_calls: dict[str, Mock], failure: str
) -> None:
    error = RuntimeError(f"startup failed during {failure}")
    if failure == "ports":
        start_calls["url"].side_effect = ["http://127.0.0.1:12345", error]
    elif failure == "readiness":
        start_calls["request"].side_effect = error
    else:
        start_calls["record_versions"].side_effect = error
    for name in ("scheduling", "evidence", "stream"):
        start_calls[name].side_effect = AssertionError(
            "Artifact endpoint used after incomplete startup."
        )

    with pytest.raises(RuntimeError, match=f"startup failed during {failure}"):
        harness.start()
    assert harness.agent_url
    harness.capture()

    assert (harness.directory / "containers.log").read_text() == "container logs\n"
    assert not (harness.directory / "artifacts.json").exists()
    for name in ("scheduling", "evidence", "stream"):
        start_calls[name].assert_not_called()


def test_completed_startup_captures_runtime_artifacts(
    harness: Runtime, start_calls: dict[str, Mock]
) -> None:
    harness.start()
    harness.capture()

    start_calls["record_versions"].assert_called_once()
    start_calls["stream"].assert_any_call("inbox")
    start_calls["stream"].assert_any_call("dlt")
    assert json.loads((harness.directory / "artifacts.json").read_text()) == {
        "scheduling": [],
        "receiver": {"calls": [], "records": []},
        "inbox": [],
        "dlt": [],
    }


def _tool_message(name: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


@pytest.mark.parametrize(
    ("current_calls", "expected_tool"),
    [
        ((), "list_drasi_subscriptions"),
        (("list_drasi_subscriptions",), "record_event"),
        (("list_drasi_subscriptions", "record_event"), None),
    ],
)
def test_scripted_model_uses_only_current_user_turn(
    monkeypatch: pytest.MonkeyPatch,
    current_calls: tuple[str, ...],
    expected_tool: str | None,
) -> None:
    monkeypatch.setattr(model_app, "_calls", [])
    task = "The same event delivered again."
    messages = [
        {"role": "user", "content": task},
        _tool_message("list_drasi_subscriptions"),
        _tool_message("record_event"),
        {"role": "assistant", "content": "Recorded."},
        {"role": "user", "content": task},
        *(_tool_message(name) for name in current_calls),
    ]
    response = model_app.completion(
        model_app.ChatRequest(
            messages=messages,
            tools=[
                {"type": "function", "function": {"name": name}}
                for name in ("list_drasi_subscriptions", "record_event")
            ],
        )
    )
    choice = response["choices"][0]
    if expected_tool is None:
        assert choice["finish_reason"] == "stop"
        assert not choice["message"].get("tool_calls")
    else:
        assert choice["finish_reason"] == "tool_calls"
        function = choice["message"]["tool_calls"][0]["function"]
        assert function["name"] == expected_tool
        assert json.loads(function["arguments"]) == (
            {"task": task} if expected_tool == "record_event" else {}
        )
