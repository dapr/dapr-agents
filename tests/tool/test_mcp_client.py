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

"""Tests for MCPClient prompt loading and accessors."""

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
    ``_server_configs``, which both modes populate.
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
