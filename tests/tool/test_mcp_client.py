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
from unittest.mock import AsyncMock

from mcp.types import Prompt

from dapr_agents.tool.mcp import MCPClient


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
