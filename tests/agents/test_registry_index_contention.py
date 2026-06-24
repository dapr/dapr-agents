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

"""
Concurrency tests for the shared team-registry index.

The team index is a single shared document that every agent in a team mutates on
register/deregister. When a whole team starts or stops at once (e.g. Ctrl-C on a
multi-agent run) all writers do a read-modify-write on the same document, so ETag
conflicts are guaranteed. These tests reproduce that contention and assert that
every writer converges — no agent is dropped from or stuck in the index.
"""

import copy
import threading

import pytest

import dapr_agents.agents.components as components
from dapr_agents.agents.components import DaprInfra, _REGISTRY_AGENTS_KEY
from dapr_agents.agents.configs import AgentRegistryConfig
from dapr_agents.storage.daprstores.stateservice import StateStoreError


class FakeEtagStore:
    """In-memory state store enforcing ETag optimistic concurrency.

    Models a Dapr state store with first-write-wins: a save whose ``etag`` does
    not match the currently stored etag raises ``StateStoreError``, exactly as the
    real store does on a conflict. Thread-safe, so it can model many agents
    mutating the shared team index simultaneously. Values are deep-copied in and
    out so callers never share mutable references outside the ETag protection.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._values = {}
        self._etags = {}
        self._version = 0
        self.deleted = []

    def seed(self, key, value):
        with self._lock:
            self._version += 1
            self._values[key] = copy.deepcopy(value)
            self._etags[key] = str(self._version)

    def load_with_etag(self, *, key, default=None, state_metadata=None):
        with self._lock:
            if key not in self._values:
                return (copy.deepcopy(default), None)
            return (copy.deepcopy(self._values[key]), self._etags[key])

    def save(self, *, key, value, etag=None, state_metadata=None, state_options=None):
        with self._lock:
            current = self._etags.get(key)
            if current is not None and etag != current:
                raise StateStoreError(
                    f"etag mismatch for {key}: stored={current} provided={etag}"
                )
            if current is None and etag is not None:
                raise StateStoreError(f"etag provided for missing key {key}")
            self._version += 1
            self._values[key] = copy.deepcopy(value)
            self._etags[key] = str(self._version)

    def delete(self, *, key, state_metadata=None, etag=None):
        with self._lock:
            self.deleted.append(key)
            self._values.pop(key, None)
            self._etags.pop(key, None)


def _make_infra(name, store):
    return DaprInfra(
        name=name,
        registry=AgentRegistryConfig(store=store, team_name="default"),
    )


@pytest.fixture
def fast_backoff(monkeypatch):
    """Shrink backoff so contention tests run quickly while still de-correlating."""
    monkeypatch.setattr(components, "_INDEX_BACKOFF_BASE_SECONDS", 0.001)
    monkeypatch.setattr(components, "_INDEX_BACKOFF_CAP_SECONDS", 0.03)


def _run_concurrently(targets):
    errors = []

    def wrap(fn):
        def inner():
            try:
                fn()
            except Exception as exc:  # pragma: no cover - failures surface via errors
                errors.append(exc)

        return inner

    threads = [threading.Thread(target=wrap(fn)) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a worker thread hung"
    return errors


@pytest.mark.parametrize("num_agents", [8])
def test_concurrent_deregister_all_converge(fast_backoff, num_agents):
    """Every agent deregisters concurrently and the index ends empty."""
    store = FakeEtagStore()
    names = [f"agent-{i}" for i in range(num_agents)]
    infras = [_make_infra(n, store) for n in names]

    index_key = infras[0]._team_registry_index_key("default")
    store.seed(index_key, {_REGISTRY_AGENTS_KEY: list(names)})

    errors = _run_concurrently([inf.deregister_agentic_system for inf in infras])

    assert not errors, f"deregistration raised under contention: {errors}"
    final, _ = store.load_with_etag(key=index_key)
    assert final[_REGISTRY_AGENTS_KEY] == [], (
        f"every agent should be removed from the index, got {final}"
    )
    # Each agent also deletes its own per-agent key (step 1).
    for n in names:
        assert f"agents:default:{n}" in store.deleted


@pytest.mark.parametrize("num_agents", [8])
def test_concurrent_index_add_all_converge(fast_backoff, num_agents):
    """Every agent adds itself to the index concurrently; none is lost."""
    store = FakeEtagStore()
    names = [f"agent-{i}" for i in range(num_agents)]
    infras = [_make_infra(n, store) for n in names]

    index_key = infras[0]._team_registry_index_key("default")
    store.seed(index_key, {_REGISTRY_AGENTS_KEY: []})
    partition = infras[0]._registry_partition_key("default")

    def adder(infra):
        def _add(agents_list):
            if infra.name in agents_list:
                return False
            agents_list.append(infra.name)
            return True

        def run():
            assert infra._mutate_team_index(
                index_key=index_key, partition_meta=partition, mutate=_add
            )

        return run

    errors = _run_concurrently([adder(inf) for inf in infras])

    assert not errors, f"index add raised under contention: {errors}"
    final, _ = store.load_with_etag(key=index_key)
    assert sorted(final[_REGISTRY_AGENTS_KEY]) == sorted(names), (
        f"all agents should be present exactly once, got {final}"
    )


def test_mutate_team_index_gives_up_when_always_conflicting(monkeypatch):
    """A permanently-conflicting store fails bounded — it must not hang."""
    monkeypatch.setattr(components, "_INDEX_MUTATION_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(components, "_INDEX_MUTATION_BUDGET_SECONDS", 5.0)
    monkeypatch.setattr(components, "_INDEX_BACKOFF_BASE_SECONDS", 0.001)
    monkeypatch.setattr(components, "_INDEX_BACKOFF_CAP_SECONDS", 0.005)

    store = FakeEtagStore()
    index_key = "agents:default:_index"
    store.seed(index_key, {_REGISTRY_AGENTS_KEY: ["agent-0"]})

    load_calls = {"n": 0}
    real_load = store.load_with_etag

    def flaky_load(**kw):
        load_calls["n"] += 1
        return real_load(**kw)

    # Every save loses the ETag race (simulates a writer that is always last).
    def always_conflict(**kw):
        raise StateStoreError("permanent conflict")

    store.load_with_etag = flaky_load
    store.save = always_conflict

    infra = _make_infra("agent-0", store)

    def _remove(agents_list):
        if "agent-0" not in agents_list:
            return False
        agents_list.remove("agent-0")
        return True

    ok = infra._mutate_team_index(
        index_key=index_key,
        partition_meta={},
        mutate=_remove,
    )

    assert ok is False
    assert load_calls["n"] == 4, "should retry up to the attempt bound, then give up"


def test_mutate_team_index_retries_then_succeeds(monkeypatch):
    """A few transient conflicts are absorbed; the write eventually lands."""
    monkeypatch.setattr(components, "_INDEX_BACKOFF_BASE_SECONDS", 0.001)
    monkeypatch.setattr(components, "_INDEX_BACKOFF_CAP_SECONDS", 0.005)

    store = FakeEtagStore()
    index_key = "agents:default:_index"
    store.seed(index_key, {_REGISTRY_AGENTS_KEY: ["agent-0"]})

    real_save = store.save
    fail_budget = {"n": 2}

    def flaky_save(**kw):
        if fail_budget["n"] > 0:
            fail_budget["n"] -= 1
            raise StateStoreError("transient conflict")
        return real_save(**kw)

    store.save = flaky_save
    infra = _make_infra("agent-0", store)

    def _remove(agents_list):
        if "agent-0" not in agents_list:
            return False
        agents_list.remove("agent-0")
        return True

    ok = infra._mutate_team_index(
        index_key=index_key, partition_meta={}, mutate=_remove
    )

    assert ok is True
    final, _ = store.load_with_etag(key=index_key)
    assert final[_REGISTRY_AGENTS_KEY] == []
