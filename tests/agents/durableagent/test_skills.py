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
import threading
from typing import Any, Sequence
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from dapr.ext.workflow import WorkflowActivityContext

from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.skills import (
    Skill,
    SkillCatalogError,
    SkillMetadata,
    SkillResource,
)
from dapr_agents.skills._durable import SkillStateStore
from dapr_agents.storage.daprstores.stateservice import StateStoreError
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from tests.agents.durableagent.test_durable_agent import MockDaprClient
from tests.fake_state_store import FakeEtagStateStore


class FakeSource:
    def __init__(self, skill: Skill) -> None:
        self.skill = skill
        self.load_calls = 0
        self.read_calls = 0

    async def resolve_snapshot(self, *, snapshot_id: str | None = None) -> str:
        expected = f"memory:{self.skill.name}"
        if snapshot_id not in (None, expected):
            raise ValueError(snapshot_id)
        return expected

    async def list_metadata(
        self, *, snapshot_id: str | None = None
    ) -> Sequence[SkillMetadata]:
        return [SkillMetadata(name=self.skill.name, description=self.skill.description)]

    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        self.load_calls += 1
        return self.skill

    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        self.read_calls += 1
        return self.skill.resources[path]


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


@pytest.fixture
def mock_llm() -> Mock:
    llm = Mock(spec=OpenAIChatClient)
    llm.generate = AsyncMock()
    llm.prompt_template = None
    llm.__class__.__name__ = "MockLLMClient"
    llm.provider = "MockOpenAIProvider"
    llm.api = "MockOpenAIAPI"
    llm.model = "gpt-4o-mock"
    return llm


@pytest.fixture
def skill() -> Skill:
    return Skill(
        name="incident-investigation",
        description="Investigate service incidents.",
        instructions="Follow the incident investigation procedure.",
        resources={
            "references/checklist.md": SkillResource(
                path="references/checklist.md",
                content="# Checklist",
                media_type="text/markdown",
            )
        },
    )


def make_agent(
    mock_llm: Mock,
    skills: Sequence[object],
    store: StateStoreService | None = None,
) -> DurableAgent:
    return DurableAgent(
        name="SkillsAgent",
        role="Skill assistant",
        goal="Use native skills",
        system_prompt="Base system instruction.",
        llm=mock_llm,
        skills=skills,
        pubsub=AgentPubSubConfig(pubsub_name="testpubsub", agent_topic="SkillsAgent"),
        state=AgentStateConfig(
            store=store or StateStoreService(store_name="teststatestore")
        ),
        execution=AgentExecutionConfig(max_iterations=2),
    )


class ConcurrentCleanupIndexStore:
    """State-store double that exposes concurrent cleanup-index updates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, dict[str, Any]] = {}
        self._versions: dict[str, int] = {}
        self._cleanup_index_reads = 0
        self._cleanup_index_barrier = threading.Barrier(2)

    def load(self, *, key: str, default: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return dict(self._values.get(key, default))

    def load_with_etag(
        self, *, key: str, default: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        if ":cleanup-index:" in key:
            with self._lock:
                should_wait = self._cleanup_index_reads < 2
                self._cleanup_index_reads += 1
            if should_wait:
                # Force both writers to use the same cleanup-index snapshot.
                # The production ETag write must resolve this deliberate race.
                self._cleanup_index_barrier.wait()
        with self._lock:
            return (
                dict(self._values.get(key, default)),
                str(self._versions[key]) if key in self._versions else None,
            )

    def save(
        self,
        *,
        key: str,
        value: dict[str, Any],
        etag: str | None = None,
        state_options: Any = None,
    ) -> None:
        with self._lock:
            current = self._versions.get(key)
            if state_options is not None:
                if etag is None and current is not None:
                    raise StateStoreError("first-write conflict")
                if etag is not None and etag != str(current):
                    raise StateStoreError("etag conflict")
            self._values[key] = dict(value)
            self._versions[key] = (current or 0) + 1

    def delete(self, *, key: str) -> None:
        with self._lock:
            self._values.pop(key, None)
            self._versions.pop(key, None)


def make_durable_skill_agent(
    mock_llm: Mock, skill: Skill
) -> tuple[FakeEtagStateStore, StateStoreService, FakeSource, DurableAgent]:
    fake_store = FakeEtagStateStore()
    store = StateStoreService(store_factory=lambda: fake_store, store_name="skills")
    source = FakeSource(skill)
    agent = make_agent(mock_llm, [source], store=store)
    agent._infra.get_state = Mock(return_value=MagicMock(messages=[]))
    return fake_store, store, source, agent


def _tool_call(name: str, arguments: dict[str, str], call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def make_skill_tool_payload(
    agent: DurableAgent, skill: Skill, catalog: list[dict[str, Any]]
) -> dict[str, Any]:
    tool_call = _tool_call("load_skill", {"name": skill.name}, "call-1")
    assert agent._skill_runtime is not None
    return {
        "instance_id": "run-1",
        "skill_catalog": catalog,
        "skill_operation_key": agent._skill_runtime.operation_key(
            "run-1", 1, 0, tool_call
        ),
        "tool_call": tool_call,
    }


def test_constructor_registers_skill_tools_without_streaming(
    mock_llm: Mock, skill: Skill
) -> None:
    agent = make_agent(mock_llm, [skill])

    assert agent.execution.streaming is False
    assert agent.tool_executor.get_tool("load_skill") is not None
    assert agent.tool_executor.get_tool("read_skill_resource") is not None


def test_constructor_rejects_bare_skill_values(mock_llm: Mock) -> None:
    with pytest.raises(TypeError, match="Skill or SkillSource"):
        make_agent(mock_llm, ["skills/incident-investigation"])


def test_constructor_rejects_skills_with_executor(mock_llm: Mock, skill: Skill) -> None:
    with pytest.raises(ValueError, match="executor"):
        DurableAgent(name="SkillsAgent", llm=None, executor=Mock(), skills=[skill])


def test_constructor_requires_state_store_for_skills(
    mock_llm: Mock, skill: Skill
) -> None:
    with pytest.raises(ValueError, match="configured state store"):
        DurableAgent(name="SkillsAgent", llm=mock_llm, skills=[skill])


def test_skill_operation_key_is_deterministic(mock_llm: Mock, skill: Skill) -> None:
    """The same durable tool invocation must reuse its replay key."""
    agent = make_agent(mock_llm, [skill])
    tool_call = _tool_call("load_skill", {"name": skill.name}, "call-1")

    assert agent._skill_runtime is not None
    first = agent._skill_runtime.operation_key("run-1", 1, 0, tool_call)
    same = agent._skill_runtime.operation_key("run-1", 1, 0, tool_call)
    different_turn = agent._skill_runtime.operation_key("run-1", 2, 0, tool_call)

    assert first == same
    assert first != different_turn


def test_skill_state_keys_are_stable() -> None:
    state = SkillStateStore("SkillsAgent", Mock())

    assert state.catalog_key("run-1") == "dapr-agents-skills:SkillsAgent:catalog:run-1"
    assert state.result_key("run-1", "operation") == (
        "dapr-agents-skills:SkillsAgent:result:run-1:operation"
    )
    assert state.cleanup_index_key("run-1") == (
        "dapr-agents-skills:SkillsAgent:cleanup-index:run-1"
    )


def test_discover_skill_catalog_runs_through_activity(
    mock_llm: Mock, skill: Skill
) -> None:
    source_skill = skill.model_copy(update={"name": "release-notes"})
    source = FakeSource(source_skill)
    agent = make_agent(mock_llm, [skill, source])

    result = agent.discover_skill_catalog(
        Mock(spec=WorkflowActivityContext), {"instance_id": "run-1"}
    )

    assert [entry["name"] for entry in result] == [
        "incident-investigation",
        "release-notes",
    ]


def test_catalog_redelivery_uses_persisted_snapshot(
    mock_llm: Mock, skill: Skill
) -> None:
    """Redelivery must not observe provider metadata changes mid-run."""
    _, _, source, agent = make_durable_skill_agent(mock_llm, skill)
    ctx = Mock(spec=WorkflowActivityContext)

    first = agent.discover_skill_catalog(ctx, {"instance_id": "run-1"})
    source.skill = skill.model_copy(update={"description": "changed"})
    second = agent.discover_skill_catalog(ctx, {"instance_id": "run-1"})

    assert second == first
    assert second[0]["description"] == "Investigate service incidents."


@pytest.mark.parametrize(
    "catalog",
    [
        None,
        ["not-an-entry"],
        [{"description": "missing name", "_source_id": "source"}],
        [{"name": 42, "description": "invalid name", "_source_id": "source"}],
        [{"name": "missing-source", "description": "missing source id"}],
    ],
)
def test_malformed_cached_catalog_fails_closed(
    mock_llm: Mock, skill: Skill, catalog: Any
) -> None:
    fake_store, _, _, agent = make_durable_skill_agent(mock_llm, skill)
    assert agent._skill_runtime is not None
    fake_store.seed(
        agent._skill_runtime._state_store.catalog_key("run-1"),
        {"catalog": catalog},
    )

    with pytest.raises(SkillCatalogError):
        agent.discover_skill_catalog(
            Mock(spec=WorkflowActivityContext), {"instance_id": "run-1"}
        )


def test_cleanup_index_updates_preserve_concurrent_result_keys() -> None:
    # SkillStateStore.save() is synchronous, so threads model two parallel
    # workflow activities.  Sequential saves would never expose a lost
    # read-modify-write update to the shared cleanup index.
    store = ConcurrentCleanupIndexStore()
    stores = [
        SkillStateStore("SkillsAgent", store),
        SkillStateStore("SkillsAgent", store),
    ]
    threads = [
        threading.Thread(
            target=state_store.save,
            args=("run-1", f"result-{index}", {"tool_result": {}}),
        )
        for index, state_store in enumerate(stores)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # This is an ordinary post-join inspection.  The synchronization barrier
    # belongs in load_with_etag(), where it coordinates the two writers; placing
    # it here would make this single final reader wait forever for a second
    # reader that does not exist.
    cleanup_index = store.load(key=stores[0].cleanup_index_key("run-1"), default={})
    assert cleanup_index["keys"] == ["result-0", "result-1"]


def test_cleanup_index_is_deleted_last_after_data_cleanup() -> None:
    class FailingDeleteStore:
        def __init__(self) -> None:
            self.deleted: list[str] = []
            self.fail_result_delete = True

        def load(self, *, key: str, default: dict[str, Any]) -> dict[str, Any]:
            if key == state_store.cleanup_index_key("run-1"):
                return {"keys": [state_store.result_key("run-1", "operation")]}
            return dict(default)

        def delete(self, *, key: str) -> None:
            self.deleted.append(key)
            if (
                key == state_store.result_key("run-1", "operation")
                and self.fail_result_delete
            ):
                raise StateStoreError("result delete failed")

    store = FailingDeleteStore()
    state_store = SkillStateStore("SkillsAgent", store)  # type: ignore[arg-type]

    state_store.clear("run-1")

    cleanup_index_key = state_store.cleanup_index_key("run-1")
    assert cleanup_index_key not in store.deleted

    store.fail_result_delete = False
    state_store.clear("run-1")

    assert cleanup_index_key in store.deleted


def test_malformed_cached_result_is_ignored(mock_llm: Mock, skill: Skill) -> None:
    fake_store, _, source, agent = make_durable_skill_agent(mock_llm, skill)
    catalog = agent.discover_skill_catalog(
        Mock(spec=WorkflowActivityContext), {"instance_id": "run-1"}
    )
    payload = make_skill_tool_payload(agent, skill, catalog)
    assert agent._skill_runtime is not None
    fake_store.seed(
        agent._skill_runtime._state_store.result_key(
            "run-1", payload["skill_operation_key"]
        ),
        {"tool_result": {"role": "tool"}},
    )

    result = agent.run_tool(Mock(spec=WorkflowActivityContext), payload)

    assert json.loads(result["content"])["name"] == skill.name
    assert source.load_calls == 1


def test_non_mapping_cached_result_is_ignored(mock_llm: Mock, skill: Skill) -> None:
    fake_store, _, source, agent = make_durable_skill_agent(mock_llm, skill)
    catalog = agent.discover_skill_catalog(
        Mock(spec=WorkflowActivityContext), {"instance_id": "run-1"}
    )
    payload = make_skill_tool_payload(agent, skill, catalog)
    assert agent._skill_runtime is not None
    fake_store.seed(
        agent._skill_runtime._state_store.result_key(
            "run-1", payload["skill_operation_key"]
        ),
        {"tool_result": "not-a-tool-message"},
    )

    result = agent.run_tool(Mock(spec=WorkflowActivityContext), payload)

    assert json.loads(result["content"])["name"] == skill.name
    assert source.load_calls == 1


def test_skill_result_redelivery_uses_persisted_result(
    mock_llm: Mock, skill: Skill
) -> None:
    _, store, source, agent = make_durable_skill_agent(mock_llm, skill)
    catalog = agent.discover_skill_catalog(
        Mock(spec=WorkflowActivityContext), {"instance_id": "run-1"}
    )
    payload = make_skill_tool_payload(agent, skill, catalog)

    first = agent.run_tool(Mock(spec=WorkflowActivityContext), payload)
    source.skill = skill.model_copy(update={"instructions": "changed"})
    restarted = make_agent(mock_llm, [source], store=store)
    restarted._infra.get_state = Mock(return_value=MagicMock(messages=[]))
    second = restarted.run_tool(Mock(spec=WorkflowActivityContext), payload)

    assert second == first
    assert json.loads(second["content"])["instructions"] == (
        "Follow the incident investigation procedure."
    )
    assert source.load_calls == 1


def test_skill_resource_result_redelivery_uses_persisted_result(
    mock_llm: Mock, skill: Skill
) -> None:
    """A retried resource activity must replay without rereading the source."""
    _, store, source, agent = make_durable_skill_agent(mock_llm, skill)
    catalog = agent.discover_skill_catalog(
        Mock(spec=WorkflowActivityContext), {"instance_id": "run-1"}
    )
    load_payload = make_skill_tool_payload(agent, skill, catalog)
    agent.run_tool(Mock(spec=WorkflowActivityContext), load_payload)

    read_call = _tool_call(
        "read_skill_resource",
        {"name": skill.name, "path": "references/checklist.md"},
        "read-1",
    )
    assert agent._skill_runtime is not None
    read_payload = {
        "instance_id": "run-1",
        "skill_catalog": catalog,
        "skill_operation_key": agent._skill_runtime.operation_key(
            "run-1", 1, 1, read_call
        ),
        "tool_call": read_call,
    }
    first = agent.run_tool(Mock(spec=WorkflowActivityContext), read_payload)
    restarted = make_agent(mock_llm, [source], store=store)
    second = restarted.run_tool(Mock(spec=WorkflowActivityContext), read_payload)

    assert second == first
    assert json.loads(second["content"])["content"] == "# Checklist"
    assert source.read_calls == 1


def test_cached_skill_error_survives_redelivery(mock_llm: Mock, skill: Skill) -> None:
    _, store, source, agent = make_durable_skill_agent(mock_llm, skill)
    source.load_skill = AsyncMock(side_effect=RuntimeError("provider secret"))
    catalog = agent.discover_skill_catalog(
        Mock(spec=WorkflowActivityContext), {"instance_id": "run-1"}
    )
    payload = make_skill_tool_payload(agent, skill, catalog)

    first = agent.run_tool(Mock(spec=WorkflowActivityContext), payload)
    source.load_skill = AsyncMock(return_value=skill)
    restarted = make_agent(mock_llm, [source], store=store)
    second = restarted.run_tool(Mock(spec=WorkflowActivityContext), payload)

    assert second == first
    assert "provider secret" not in first["content"]
    source.load_skill.assert_not_awaited()


def test_unexpected_skill_tool_failure_is_redacted(
    mock_llm: Mock, skill: Skill
) -> None:
    _, _, _, agent = make_durable_skill_agent(mock_llm, skill)
    catalog = agent.discover_skill_catalog(
        Mock(spec=WorkflowActivityContext), {"instance_id": "run-1"}
    )
    assert agent._skill_runtime is not None
    agent._skill_runtime._runtime.load = AsyncMock(
        side_effect=RuntimeError("credential=secret-token")
    )

    result = agent.run_tool(
        Mock(spec=WorkflowActivityContext),
        make_skill_tool_payload(agent, skill, catalog),
    )

    assert "access error" in result["content"]
    assert "secret-token" not in result["content"]


def test_finalization_clears_durable_skill_state(mock_llm: Mock, skill: Skill) -> None:
    fake_store, store, source, agent = make_durable_skill_agent(mock_llm, skill)
    catalog = agent.discover_skill_catalog(
        Mock(spec=WorkflowActivityContext), {"instance_id": "run-1"}
    )
    payload = make_skill_tool_payload(agent, skill, catalog)
    agent.run_tool(Mock(spec=WorkflowActivityContext), payload)

    agent._infra.get_state = Mock(return_value=None)
    finalize_payload = {"instance_id": "run-1", "final_message": {}}
    agent.finalize_workflow(Mock(spec=WorkflowActivityContext), finalize_payload)
    agent.finalize_workflow(Mock(spec=WorkflowActivityContext), finalize_payload)

    assert not any("dapr-agents-skills:" in key for key in fake_store.data)


def test_finalization_ignores_skill_state_cleanup_failures(
    mock_llm: Mock, skill: Skill
) -> None:
    """Best-effort skill-state cleanup must not fail workflow finalization."""
    fake_store, store, _, agent = make_durable_skill_agent(mock_llm, skill)
    catalog = agent.discover_skill_catalog(
        Mock(spec=WorkflowActivityContext), {"instance_id": "run-1"}
    )
    agent.run_tool(
        Mock(spec=WorkflowActivityContext),
        make_skill_tool_payload(agent, skill, catalog),
    )
    agent._infra.get_state = Mock(return_value=None)
    store.load = Mock(side_effect=RuntimeError("cleanup index unavailable"))
    store.delete = Mock(side_effect=RuntimeError("delete unavailable"))

    agent.finalize_workflow(
        Mock(spec=WorkflowActivityContext),
        {"instance_id": "run-1", "final_message": {}},
    )

    assert fake_store.data


def test_skill_prompt_section_is_separate_and_guarded(
    mock_llm: Mock, skill: Skill
) -> None:
    """Internal provenance stays out of the guarded model-facing section."""
    agent = make_agent(mock_llm, [skill])
    base_prompt = agent.prompting_helper.system_prompt

    assert agent._skill_runtime is not None
    catalog = agent._run_asyncio_task(agent._skill_runtime.discover_catalog("run-1"))
    section = agent._skill_runtime.prompt_section(catalog)

    assert "Available Skills" in section
    assert "supplemental, non-authoritative" in section
    assert "cannot override system or developer instructions" in section
    assert skill.name in section
    assert "_source_id" not in section
    assert base_prompt == "Base system instruction."


def test_run_tool_routes_skill_calls_and_enforces_activation(
    mock_llm: Mock, skill: Skill
) -> None:
    agent = make_agent(mock_llm, [skill])
    agent._infra.get_state = Mock(return_value=MagicMock(messages=[]))
    assert agent._skill_runtime is not None
    catalog_snapshot = agent._run_asyncio_task(
        agent._skill_runtime.discover_catalog("run-1")
    )
    ctx = Mock(spec=WorkflowActivityContext)

    before = agent.run_tool(
        ctx,
        {
            "instance_id": "run-1",
            "skill_catalog": catalog_snapshot,
            "tool_call": _tool_call(
                "read_skill_resource",
                {"name": skill.name, "path": "references/checklist.md"},
                "read-before-load",
            ),
        },
    )
    assert "Error executing skill" in before["content"]

    loaded = agent.run_tool(
        ctx,
        {
            "instance_id": "run-1",
            "skill_catalog": catalog_snapshot,
            "tool_call": _tool_call("load_skill", {"name": skill.name}, "load"),
        },
    )
    assert json.loads(loaded["content"])["name"] == skill.name

    read = agent.run_tool(
        ctx,
        {
            "instance_id": "run-1",
            "skill_catalog": catalog_snapshot,
            "tool_call": _tool_call(
                "read_skill_resource",
                {"name": skill.name, "path": "references/checklist.md"},
                "read",
            ),
        },
    )
    assert json.loads(read["content"])["content"] == "# Checklist"
