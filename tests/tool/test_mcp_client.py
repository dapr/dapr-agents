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

"""Tests for MCPClient connect/close behavior and prompt accessors."""

import asyncio
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import Prompt

from dapr_agents.tool.mcp import MCPClient


def _make_session():
    return SimpleNamespace(
        initialize=AsyncMock(),
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])),
        list_prompts=AsyncMock(return_value=SimpleNamespace(prompts=[])),
    )


async def test_load_prompts_failure_leaves_accessors_usable():
    """A server without a prompts capability must not poison the prompt accessors.

    ``_load_prompts_from_session`` is called on every connect. When
    ``session.list_prompts()`` raises (e.g. tool-only servers that do not
    implement the prompts capability), the except branch must store an empty
    mapping so the dict-based accessors keep working. Storing a list instead
    made every accessor raise ``AttributeError`` (``list`` has no ``.values()``
    /``.keys()``/``.get()``).
    """
    client = MCPClient()
    session = SimpleNamespace(
        list_prompts=AsyncMock(side_effect=RuntimeError("prompts not supported"))
    )

    await client._load_prompts_from_session("srv", session)

    assert client.get_server_prompts("srv") == []
    assert client.get_all_prompts() == {"srv": []}
    assert client.get_prompt_names("srv") == []
    assert client.get_all_prompt_names() == {"srv": []}
    assert client.get_prompt_metadata("srv", "anything") is None


async def test_load_prompts_success_populates_accessors():
    """The success path stores the prompts keyed by name and stays readable."""
    client = MCPClient()
    prompt = Prompt(name="greet")
    session = SimpleNamespace(
        list_prompts=AsyncMock(return_value=SimpleNamespace(prompts=[prompt]))
    )

    await client._load_prompts_from_session("srv", session)

    assert client.get_prompt_names("srv") == ["greet"]
    assert client.get_server_prompts("srv") == [prompt]
    assert client.get_prompt_metadata("srv", "greet") is prompt
    assert client.get_all_prompts() == {"srv": [prompt]}


async def test_connect_ephemeral_rejects_duplicate_server_name():
    """The duplicate-connection guard must fire in ephemeral mode too.

    Ephemeral sessions (the default, ``persistent_connections=False``) never
    populate ``_sessions``, so a guard keyed on ``_sessions`` can never catch
    a duplicate ``connect()`` call in the common case. It must key on
    state that both modes populate.
    """
    client = MCPClient()
    with patch(
        "dapr_agents.tool.mcp.client.start_transport_session",
        AsyncMock(return_value=_make_session()),
    ):
        await client.connect(
            {"server_name": "srv", "transport": "stdio", "command": "python"}
        )

        with pytest.raises(RuntimeError, match="already connected"):
            await client.connect(
                {"server_name": "srv", "transport": "stdio", "command": "python"}
            )


async def test_close_allows_reconnecting_to_the_same_server():
    """close() must fully release a server so it can be connected to again."""
    client = MCPClient()
    with patch(
        "dapr_agents.tool.mcp.client.start_transport_session",
        AsyncMock(return_value=_make_session()),
    ):
        await client.connect(
            {"server_name": "srv", "transport": "stdio", "command": "python"}
        )
        await client.close()

        assert client.get_connected_servers() == []
        assert client.get_all_prompts() == {}

        await client.connect(
            {"server_name": "srv", "transport": "stdio", "command": "python"}
        )

    assert client.get_connected_servers() == ["srv"]


async def test_concurrent_connect_to_same_server_is_refused():
    """Two connects racing for one name must not both open a session."""
    client = MCPClient()
    release = asyncio.Event()

    async def slow_session(*_args, **_kwargs):
        await release.wait()
        return _make_session()

    config = {"server_name": "srv", "transport": "stdio", "command": "python"}
    with patch(
        "dapr_agents.tool.mcp.client.start_transport_session",
        AsyncMock(side_effect=slow_session),
    ) as start:
        first = asyncio.create_task(client.connect(config))
        second = asyncio.create_task(client.connect(config))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)

    assert results[0] is None
    assert isinstance(results[1], RuntimeError)
    assert "already connecting" in str(results[1])
    assert start.await_count == 1
    assert client.get_connected_servers() == ["srv"]


async def test_failed_connect_can_be_retried():
    client = MCPClient()
    config = {"server_name": "srv", "transport": "stdio", "command": "python"}
    with patch(
        "dapr_agents.tool.mcp.client.start_transport_session",
        AsyncMock(side_effect=[OSError("refused"), _make_session()]),
    ):
        with pytest.raises(OSError):
            await client.connect(config)
        assert client._server_configs == {}

        await client.connect(config)

    assert client.get_connected_servers() == ["srv"]


async def test_failed_reconnect_keeps_previous_config_and_warns(caplog):
    client = MCPClient()
    old = {"server_name": "srv", "transport": "stdio", "command": "old"}
    new = {"server_name": "srv", "transport": "stdio", "command": "new"}
    with patch(
        "dapr_agents.tool.mcp.client.start_transport_session",
        AsyncMock(side_effect=[_make_session(), OSError("refused")]),
    ):
        await client.connect(old)
        await client.close()
        with caplog.at_level(logging.WARNING, logger="dapr_agents.tool.mcp.client"):
            with pytest.raises(OSError):
                await client.connect(new)

    assert client._server_configs["srv"]["params"] == {"command": "old"}
    assert client.get_connected_servers() == []
    assert any(
        "keeping its previous configuration" in r.getMessage() for r in caplog.records
    )


# --- Real-server tests: the documented connect -> get_all_tools -> close ->
# call-a-tool pattern (see examples/06-agent-mcp-client-*). ---

_ECHO_SERVER = str(Path(__file__).with_name("mcp_echo_server.py"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def streamable_http_url():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, _ECHO_SERVER, "streamable-http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            if proc.poll() is not None:
                pytest.fail("streamable-http MCP test server exited early")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    pytest.fail("streamable-http MCP test server did not start")
                time.sleep(0.1)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


async def _close_ignoring_cancel_scope(client: MCPClient) -> None:
    # Same tolerance as the examples: stdio teardown can trip anyio's
    # cancel-scope check, which does not affect the stored configs.
    try:
        await client.close()
    except RuntimeError as exc:
        if "Attempted to exit cancel scope" not in str(exc):
            raise


def _tool_text(result) -> str:
    assert not result.isError, result.content
    return result.content[0].text


async def test_stdio_tools_still_work_after_close():
    """Tools from get_all_tools() must keep working after client.close().

    Ephemeral tool calls open a new session from the stored server config, so
    close() must not drop that config.
    """
    client = MCPClient()
    await client.connect_stdio(
        server_name="echo", command=sys.executable, args=[_ECHO_SERVER, "stdio"]
    )
    tools = {tool.name: tool for tool in client.get_all_tools()}
    await _close_ignoring_cancel_scope(client)

    result = await tools["echo_add"].arun(a=2, b=3)

    assert _tool_text(result) == "sum=5"


async def test_streamable_http_tools_still_work_after_close(streamable_http_url):
    client = MCPClient()
    await client.connect_streamable_http(server_name="echo", url=streamable_http_url)
    tools = {tool.name: tool for tool in client.get_all_tools()}
    await client.close()

    result = await tools["echo_add"].arun(a=4, b=5)

    assert _tool_text(result) == "sum=9"


async def test_stdio_duplicate_connect_refused_until_close():
    client = MCPClient()
    await client.connect_stdio(
        server_name="echo", command=sys.executable, args=[_ECHO_SERVER, "stdio"]
    )
    with pytest.raises(RuntimeError, match="already connected"):
        await client.connect_stdio(
            server_name="echo", command=sys.executable, args=[_ECHO_SERVER, "stdio"]
        )

    await _close_ignoring_cancel_scope(client)
    assert client.get_connected_servers() == []

    await client.connect_stdio(
        server_name="echo", command=sys.executable, args=[_ECHO_SERVER, "stdio"]
    )
    assert client.get_connected_servers() == ["echo"]
    tools = {tool.name: tool for tool in client.get_all_tools()}
    assert _tool_text(await tools["echo_add"].arun(a=1, b=1)) == "sum=2"
    await _close_ignoring_cancel_scope(client)
