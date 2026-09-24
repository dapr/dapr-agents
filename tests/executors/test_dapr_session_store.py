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

"""Tests for the Dapr state-store-backed Claude session store."""

import json

import pytest

from dapr_agents.agents.executors import DaprSessionStore, DaprSessionStoreConfig
from dapr_agents.agents.executors import dapr_session_store as store_module
from dapr_agents.storage.daprstores.stateservice import StateStoreError
from tests.executors.fake_state import FakeStateStore

KEY = {"project_key": "proj", "session_id": "sess"}


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(store_module, "_backoff", lambda attempt: None)


def _store(state: FakeStateStore, **config) -> DaprSessionStore:
    return DaprSessionStore(state, config=DaprSessionStoreConfig(**config))


def _uuids(entries):
    return [e.get("uuid") for e in entries]


class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_chunk_bytes": 0},
            {"dedupe_window": -1},
            {"max_commit_attempts": 0},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            DaprSessionStoreConfig(**kwargs)

    def test_default_config(self):
        store = DaprSessionStore(FakeStateStore())
        assert store.config == DaprSessionStoreConfig()


class TestAppendAndLoad:
    async def test_load_of_missing_key_is_none(self):
        assert await _store(FakeStateStore()).load(KEY) is None

    async def test_empty_append_writes_nothing(self):
        state = FakeStateStore()
        await _store(state).append(KEY, [])
        assert state.data == {}

    async def test_preserves_append_order(self):
        store = _store(FakeStateStore())
        await store.append(KEY, [{"uuid": "a"}, {"uuid": "b"}])
        await store.append(KEY, [{"uuid": "c"}])
        assert _uuids(await store.load(KEY)) == ["a", "b", "c"]

    async def test_dedupes_by_uuid_and_keeps_uuidless_entries(self):
        store = _store(FakeStateStore())
        await store.append(KEY, [{"uuid": "a"}, {"type": "cost-state"}])
        await store.append(
            KEY, [{"uuid": "a"}, {"type": "cost-state"}, {"uuid": "b"}, {"uuid": "b"}]
        )
        loaded = await store.load(KEY)
        assert _uuids(loaded) == ["a", None, None, "b"]

    async def test_fully_duplicate_append_writes_no_chunk(self):
        state = FakeStateStore()
        store = _store(state)
        await store.append(KEY, [{"uuid": "a"}])
        keys_before = set(state.data)
        await store.append(KEY, [{"uuid": "a"}])
        assert set(state.data) == keys_before

    async def test_dedupe_window_bounds_memory(self):
        state = FakeStateStore()
        store = _store(state, dedupe_window=2)
        await store.append(KEY, [{"uuid": "a"}, {"uuid": "b"}, {"uuid": "c"}])
        manifest = json.loads(state.data["claude-session:proj/sess"])
        assert manifest["recent_uuids"] == ["b", "c"]
        # ``a`` fell out of the window, so it is accepted again.
        await store.append(KEY, [{"uuid": "a"}])
        assert _uuids(await store.load(KEY)) == ["a", "b", "c", "a"]

    async def test_zero_dedupe_window_disables_dedupe(self):
        store = _store(FakeStateStore(), dedupe_window=0)
        await store.append(KEY, [{"uuid": "a"}])
        await store.append(KEY, [{"uuid": "a"}])
        assert _uuids(await store.load(KEY)) == ["a", "a"]

    async def test_chunks_large_appends(self):
        state = FakeStateStore()
        store = _store(state, max_chunk_bytes=100)
        entries = [{"uuid": str(i), "text": "x" * 60} for i in range(4)]
        await store.append(KEY, entries)
        manifest = json.loads(state.data["claude-session:proj/sess"])
        assert len(manifest["chunks"]) == 4
        assert await store.load(KEY) == entries

    async def test_oversized_entry_gets_own_chunk(self):
        state = FakeStateStore()
        store = _store(state, max_chunk_bytes=10)
        big = {"uuid": "big", "text": "y" * 100}
        await store.append(KEY, [big])
        assert await store.load(KEY) == [big]

    async def test_manifest_without_chunks_loads_empty(self):
        state = FakeStateStore()
        await _store(state).append({**KEY, "subpath": "sub/a"}, [{"uuid": "z"}])
        # Main manifest exists (it lists the subkey) but holds no entries.
        assert await _store(state).load(KEY) == []

    async def test_missing_chunk_refuses_to_load(self):
        state = FakeStateStore()
        store = _store(state)
        await store.append(KEY, [{"uuid": "a"}])
        chunk_key = next(k for k in state.data if "#" in k)
        del state.data[chunk_key]
        with pytest.raises(StateStoreError, match="missing"):
            await store.load(KEY)

    async def test_key_prefix_and_reserved_separator(self):
        state = FakeStateStore()
        store = _store(state, key_prefix="p:")
        await store.append({"project_key": "a||b", "session_id": "s"}, [{"x": 1}])
        assert "p:a__b/s" in state.data

    async def test_ttl_is_applied_to_every_write(self):
        state = FakeStateStore()
        await _store(state, ttl_in_seconds=60).append(KEY, [{"uuid": "a"}])
        assert state.saves and all(s["ttl"] == 60 for s in state.saves)


class TestConcurrency:
    async def test_retries_after_etag_conflict_and_cleans_orphans(self):
        state = FakeStateStore()
        store = _store(state)
        await store.append(KEY, [{"uuid": "a"}])
        state.conflicts_to_inject = 1
        await store.append(KEY, [{"uuid": "b"}])
        assert _uuids(await store.load(KEY)) == ["a", "b"]
        manifest = json.loads(state.data["claude-session:proj/sess"])
        chunk_keys = {k for k in state.data if "#" in k}
        # The losing attempt's chunk was deleted; only referenced chunks remain.
        assert len(chunk_keys) == len(manifest["chunks"]) == 2

    async def test_first_write_conflict_is_retried(self):
        state = FakeStateStore()
        state.conflicts_to_inject = 1
        store = _store(state)
        await store.append(KEY, [{"uuid": "a"}])
        assert _uuids(await store.load(KEY)) == ["a"]

    async def test_gives_up_after_max_attempts(self):
        state = FakeStateStore()
        store = _store(state, max_commit_attempts=2)
        await store.append(KEY, [{"uuid": "a"}])
        state.conflicts_to_inject = 5
        with pytest.raises(StateStoreError):
            await store.append(KEY, [{"uuid": "b"}])
        assert _uuids(await store.load(KEY)) == ["a"]

    async def test_orphan_delete_failure_is_not_fatal(self):
        state = FakeStateStore()
        store = _store(state)
        await store.append(KEY, [{"uuid": "a"}])
        state.conflicts_to_inject = 1
        state.fail_deletes = True
        await store.append(KEY, [{"uuid": "b"}])
        assert _uuids(await store.load(KEY)) == ["a", "b"]


class TestSubkeysAndDelete:
    async def test_list_subkeys(self):
        store = _store(FakeStateStore())
        assert await store.list_subkeys(KEY) == []
        await store.append({**KEY, "subpath": "subagents/a"}, [{"uuid": "1"}])
        await store.append({**KEY, "subpath": "subagents/a"}, [{"uuid": "2"}])
        await store.append({**KEY, "subpath": "subagents/b"}, [{"uuid": "3"}])
        assert await store.list_subkeys(KEY) == ["subagents/a", "subagents/b"]
        assert _uuids(await store.load({**KEY, "subpath": "subagents/a"})) == [
            "1",
            "2",
        ]

    async def test_subkey_registration_retries_conflicts(self):
        state = FakeStateStore()
        store = _store(state)
        await store.append(KEY, [{"uuid": "a"}])
        # First conflict hits the sub transcript, second the main manifest.
        state.conflicts_to_inject = 2
        await store.append({**KEY, "subpath": "s"}, [{"uuid": "z"}])
        assert await store.list_subkeys(KEY) == ["s"]

    async def test_delete_cascades_to_subkeys(self):
        state = FakeStateStore()
        store = _store(state)
        await store.append(KEY, [{"uuid": "a"}])
        await store.append({**KEY, "subpath": "s"}, [{"uuid": "z"}])
        await store.delete(KEY)
        assert state.data == {}
        assert await store.load(KEY) is None

    async def test_delete_subkey_only(self):
        state = FakeStateStore()
        store = _store(state)
        await store.append(KEY, [{"uuid": "a"}])
        await store.append({**KEY, "subpath": "s"}, [{"uuid": "z"}])
        await store.delete({**KEY, "subpath": "s"})
        assert await store.load({**KEY, "subpath": "s"}) is None
        assert _uuids(await store.load(KEY)) == ["a"]

    async def test_delete_missing_is_noop(self):
        await _store(FakeStateStore()).delete(KEY)
