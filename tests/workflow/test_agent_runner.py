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

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dapr_agents.workflow.runners.agent import AgentRunner

# ---------------------------------------------------------------------------
# AgentRunner health endpoint checks
# ---------------------------------------------------------------------------


def _make_agent_runner() -> AgentRunner:
    return AgentRunner(name="test-agent-runner", wf_client=MagicMock())


def _make_mock_agent(is_started: bool, pubsub: Any = None) -> MagicMock:
    agent = MagicMock()
    agent.name = "mock-agent"
    agent.is_started = is_started

    if pubsub is None:
        agent.pubsub = None
    else:
        agent.pubsub = MagicMock()

    return agent


@dataclass
class _MockResponse:
    status_code: int
    detail: Any

    def json(self) -> Any:
        return self.detail


class _MockFastAPI:
    def __init__(self) -> None:
        self.routes: list[dict[str, Any]] = []

    def add_api_route(
        self,
        path: str,
        endpoint: Any,
        methods: list[str],
        summary: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.routes.append(
            {
                "path": path,
                "endpoint": endpoint,
                "methods": set(methods),
                "summary": summary,
                "tags": tags or [],
            }
        )


@pytest.mark.asyncio
class _MockClient:
    def __init__(self, app: _MockFastAPI) -> None:
        self.app = app

    async def get(self, path: str) -> _MockResponse:
        route = next(
            (r for r in self.app.routes if r["path"] == path and "GET" in r["methods"]),
            None,
        )
        if route is None:
            return _MockResponse(status_code=404, detail={"detail": "Not Found"})

        endpoint = route["endpoint"]

        try:
            result = await endpoint()
            return _MockResponse(status_code=200, detail=result)
        except Exception as exc:  # noqa: BLE001
            status_code = int(getattr(exc, "status_code", 500))
            detail = getattr(exc, "detail", "Internal Server Error")
            return _MockResponse(status_code=status_code, detail={"detail": detail})


def _make_mock_fastapi_app() -> _MockFastAPI:
    return _MockFastAPI()


@pytest.mark.asyncio
async def test_livez_ok():
    """When the agent is started, livez reports 'ok'."""
    runner = _make_agent_runner()
    mock_fastapi_app = _make_mock_fastapi_app()
    agent = _make_mock_agent(is_started=True)

    with patch(
        "dapr_agents.workflow.runners.agent.register_http_routes",
        return_value=([MagicMock()], [MagicMock()]),
    ):
        runner.serve(agent, app=mock_fastapi_app, expose_entry=False)

    client = _MockClient(mock_fastapi_app)
    livez = await client.get("/livez")

    assert livez.status_code == 200


@pytest.mark.asyncio
async def test_readyz_ok():
    """When the agent is started, readyz reports 'ready'."""
    runner = _make_agent_runner()
    mock_fastapi_app = _make_mock_fastapi_app()
    agent = _make_mock_agent(is_started=True)

    with patch(
        "dapr_agents.workflow.runners.agent.register_http_routes",
        return_value=([MagicMock()], [MagicMock()]),
    ):
        runner.serve(agent, app=mock_fastapi_app, expose_entry=False)

    client = _MockClient(mock_fastapi_app)
    readyz = await client.get("/readyz")

    assert readyz.status_code == 200


@pytest.mark.asyncio
async def test_readyz_agent_not_started():
    """When the agent is not (yet) started, readyz reports 'not ready'."""
    runner = _make_agent_runner()
    mock_fastapi_app = _make_mock_fastapi_app()
    agent = _make_mock_agent(is_started=False)

    with patch(
        "dapr_agents.workflow.runners.agent.register_http_routes",
        return_value=([MagicMock()], [MagicMock()]),
    ):
        runner.serve(agent, app=mock_fastapi_app, expose_entry=False)

    client = _MockClient(mock_fastapi_app)
    readyz = await client.get("/readyz")

    assert readyz.status_code == 503


@pytest.mark.asyncio
async def test_readyz_agent_pubsub_not_wired():
    """When pub/sub routes are not (yet) wired, readyz reports 'not ready'."""
    runner = _make_agent_runner()
    mock_fastapi_app = _make_mock_fastapi_app()
    agent = _make_mock_agent(is_started=True, pubsub=object())

    with (
        patch(
            "dapr_agents.workflow.runners.agent.register_http_routes",
            return_value=([MagicMock()], [MagicMock()]),
        ),
        patch(
            "dapr_agents.workflow.runners.agent.register_message_routes",
            return_value=([MagicMock()], [MagicMock()]),
        ),
        patch.object(runner, "_build_pubsub_specs", return_value=[]),
    ):
        runner.serve(agent, app=mock_fastapi_app, expose_entry=False)

    client = _MockClient(mock_fastapi_app)

    readyz = await client.get("/readyz")
    assert readyz.status_code == 503


@pytest.mark.asyncio
async def test_readyz_agent_pubsub_wired_and_ready():
    """When pub/sub consumers are ready, readyz reports 'ready'."""
    runner = _make_agent_runner()
    mock_fastapi_app = _make_mock_fastapi_app()
    agent = _make_mock_agent(is_started=True, pubsub=object())

    with (
        patch(
            "dapr_agents.workflow.runners.agent.register_http_routes",
            return_value=([MagicMock()], [MagicMock()]),
        ),
        patch(
            "dapr_agents.workflow.runners.agent.register_message_routes",
            return_value=([MagicMock()], [lambda: True]),
        ),
        patch.object(runner, "_build_pubsub_specs", return_value=[MagicMock()]),
    ):
        runner.serve(agent, app=mock_fastapi_app, expose_entry=False)

    client = _MockClient(mock_fastapi_app)

    readyz = await client.get("/readyz")
    assert readyz.status_code == 200


@pytest.mark.asyncio
async def test_readyz_agent_pubsub_wired_but_not_ready():
    """When pub/sub consumers are not ready, readyz reports 'not ready'."""
    runner = _make_agent_runner()
    mock_fastapi_app = _make_mock_fastapi_app()
    agent = _make_mock_agent(is_started=True, pubsub=object())

    with (
        patch(
            "dapr_agents.workflow.runners.agent.register_http_routes",
            return_value=([MagicMock()], [MagicMock()]),
        ),
        patch(
            "dapr_agents.workflow.runners.agent.register_message_routes",
            return_value=([MagicMock()], [lambda: False]),
        ),
        patch.object(runner, "_build_pubsub_specs", return_value=[MagicMock()]),
    ):
        runner.serve(agent, app=mock_fastapi_app, expose_entry=False)

    client = _MockClient(mock_fastapi_app)

    readyz = await client.get("/readyz")
    assert readyz.status_code == 503


@pytest.mark.asyncio
async def test_readyz_agent_service_routes_mounted():
    """When default service routes are mounted, readyz reports 'ready'."""
    runner = _make_agent_runner()
    mock_fastapi_app = _make_mock_fastapi_app()
    agent = _make_mock_agent(is_started=True)

    with patch(
        "dapr_agents.workflow.runners.agent.register_http_routes",
        return_value=([MagicMock()], [MagicMock()]),
    ):
        runner.serve(agent, app=mock_fastapi_app, expose_entry=True)

    client = _MockClient(mock_fastapi_app)

    readyz = await client.get("/readyz")
    assert readyz.status_code == 200


@pytest.mark.asyncio
async def test_readyz_agent_service_routes_not_mounted():
    """When default service routes are not (yet) mounted, readyz reports 'not ready'."""
    runner = _make_agent_runner()
    mock_fastapi_app = _make_mock_fastapi_app()
    agent = _make_mock_agent(is_started=True)

    with patch(
        "dapr_agents.workflow.runners.agent.register_http_routes",
        return_value=([MagicMock()], [MagicMock()]),
    ):
        runner.serve(agent, app=mock_fastapi_app)

    runner._default_http_paths.clear()

    client = _MockClient(mock_fastapi_app)

    readyz_not_ready = await client.get("/readyz")
    assert readyz_not_ready.status_code == 503


@pytest.mark.asyncio
async def test_readyz_agent_shutting_down():
    """When shutdown is in progress, readyz reports 'not ready'."""
    runner = _make_agent_runner()
    mock_fastapi_app = _make_mock_fastapi_app()
    agent = _make_mock_agent(is_started=True)

    with patch(
        "dapr_agents.workflow.runners.agent.register_http_routes",
        return_value=([MagicMock()], [MagicMock()]),
    ):
        runner.serve(agent, app=mock_fastapi_app, expose_entry=False)

    runner.install_signal_handlers()
    try:
        runner._shutdown_event.set()
        
        client = _MockClient(mock_fastapi_app)

        readyz = await client.get("/readyz")
        assert readyz.status_code == 503
    finally:
        runner.remove_signal_handlers()
