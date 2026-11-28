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

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from dapr.ext.workflow import WorkflowActivityContext

from dapr_agents.agents.configs import AgentExecutionConfig, AgentStateConfig
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.skills import FileSystemSkillSource, SkillCatalogError
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from tests.agents.durableagent.test_durable_agent import MockDaprClient
from tests.fake_state_store import FakeEtagStateStore


def _write_skill(root: Path, name: str, description: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nUse the reference.",
        encoding="utf-8",
    )
    return skill_dir


def _tool_call(name: str, arguments: dict[str, str], call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _agent(
    source: FileSystemSkillSource, fake_store: FakeEtagStateStore
) -> DurableAgent:
    llm = Mock(spec=OpenAIChatClient)
    llm.generate = AsyncMock()
    llm.prompt_template = None
    llm.__class__.__name__ = "MockLLMClient"
    llm.provider = "MockOpenAIProvider"
    llm.api = "MockOpenAIAPI"
    llm.model = "gpt-4o-mock"
    agent = DurableAgent(
        name="FilesystemSkillsAgent",
        role="Skill assistant",
        goal="Use filesystem skills",
        llm=llm,
        skills=[source],
        state=AgentStateConfig(
            store=StateStoreService(
                store_factory=lambda: fake_store,
                store_name="filesystem-skills",
            )
        ),
        execution=AgentExecutionConfig(max_iterations=2),
    )
    agent._infra.get_state = Mock(return_value=MagicMock(messages=[]))
    return agent


@pytest.fixture(autouse=True)
def dapr_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    client = MockDaprClient()
    monkeypatch.setattr("dapr.clients.DaprClient", lambda: client)
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.base.default_dapr_client_factory",
        lambda: client,
    )
    yield
    os.environ.pop("OPENAI_API_KEY", None)


def test_filesystem_source_works_through_durable_catalog_and_tools(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    skill_dir = _write_skill(root, "incident-response", "Respond to incidents.")
    resource = skill_dir / "references" / "runbook.md"
    resource.parent.mkdir()
    resource.write_text("initial", encoding="utf-8")
    source = FileSystemSkillSource([root])
    agent = _agent(source, FakeEtagStateStore())
    context = Mock(spec=WorkflowActivityContext)

    first_catalog = agent.discover_skill_catalog(context, {"instance_id": "run-1"})
    load_call = _tool_call("load_skill", {"name": "incident-response"}, "load-1")
    loaded = agent.run_tool(
        context,
        {
            "instance_id": "run-1",
            "skill_catalog": first_catalog,
            "tool_call": load_call,
            "turn": 1,
            "order": 0,
        },
    )

    assert json.loads(loaded["content"]) == {
        "name": "incident-response",
        "instructions": "Use the reference.",
        "available_resources": ["references/runbook.md"],
    }

    resource.write_text("changed after activation", encoding="utf-8")
    read_call = _tool_call(
        "read_skill_resource",
        {"name": "incident-response", "path": "references/runbook.md"},
        "read-1",
    )
    read = agent.run_tool(
        context,
        {
            "instance_id": "run-1",
            "skill_catalog": first_catalog,
            "tool_call": read_call,
            "turn": 2,
            "order": 0,
        },
    )

    assert "Skill access error" in read["content"]

    (skill_dir / "SKILL.md").write_text(
        "---\nname: incident-response\ndescription: Updated skill.\n---\n\nNew instructions.",
        encoding="utf-8",
    )
    second_catalog = agent.discover_skill_catalog(context, {"instance_id": "run-2"})

    assert second_catalog[0]["description"] == "Updated skill."


def test_duplicate_names_fail_through_durable_catalog(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_skill(first, "same-skill", "First.")
    _write_skill(second, "same-skill", "Second.")
    agent = _agent(FileSystemSkillSource([first, second]), FakeEtagStateStore())

    with pytest.raises(SkillCatalogError):
        agent.discover_skill_catalog(
            Mock(spec=WorkflowActivityContext), {"instance_id": "run-1"}
        )
