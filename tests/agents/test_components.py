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

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generator, Optional, Tuple
from unittest.mock import Mock, patch

import pytest

from dapr_agents.agents.components import DaprInfra, _ETAG_CACHE_MAXSIZE
from dapr_agents.agents.configs import AgentStateConfig


def _load_with_etag(
    *, key: str, default: Optional[Dict[str, Any]] = None, **_: Any
) -> Tuple[Dict[str, Any], str]:
    payload = default if isinstance(default, dict) else {}
    return payload, f"etag-{key}"


@pytest.fixture
def infra() -> DaprInfra:
    store = Mock()
    store.store_name = "state"
    return DaprInfra(name="tester", state=AgentStateConfig(store=store))


@pytest.fixture
def mock_load_with_etag(infra: DaprInfra) -> Generator[Mock, None, None]:
    with patch.object(
        infra.state_store, "load_with_etag", side_effect=_load_with_etag
    ) as mocked:
        yield mocked


def _cache_key(infra: DaprInfra, instance_id: str) -> str:
    return f"{infra.state_key_prefix}_{instance_id}".lower()


def test_concurrent_get_state_keeps_etag_cache_bounded(
    infra: DaprInfra, mock_load_with_etag: Mock
) -> None:
    """Read-only get_state calls must not accumulate one etag entry per instance.

    get_state populates the per-instance etag cache, but read-only callers never
    pair it with a save_state (which is the only path that drains an entry).  On
    a long-lived DaprInfra this would grow one entry per workflow_instance_id
    forever, so the cache is bounded with LRU eviction.
    """
    total = _ETAG_CACHE_MAXSIZE + 500
    with ThreadPoolExecutor(max_workers=16) as executor:
        states = executor.map(infra.get_state, (f"instance-{i}" for i in range(total)))
        list(states)

    assert len(infra._etag_cache) <= _ETAG_CACHE_MAXSIZE


def test_etag_cache_evicts_least_recently_used(
    infra: DaprInfra, mock_load_with_etag: Mock
) -> None:
    """When full, the cache evicts the oldest instance, keeping the newest."""
    infra.get_state("first")
    for i in range(_ETAG_CACHE_MAXSIZE):
        infra.get_state(f"filler-{i}")

    assert len(infra._etag_cache) == _ETAG_CACHE_MAXSIZE
    assert _cache_key(infra, "first") not in infra._etag_cache
    assert _cache_key(infra, f"filler-{_ETAG_CACHE_MAXSIZE - 1}") in infra._etag_cache


def test_purge_state_evicts_cached_etag(
    infra: DaprInfra, mock_load_with_etag: Mock
) -> None:
    """purge_state drops any cached etag so teardown reclaims the slot."""
    infra.get_state("gone")
    assert _cache_key(infra, "gone") in infra._etag_cache

    infra.purge_state("gone")
    assert _cache_key(infra, "gone") not in infra._etag_cache


def test_purge_state_evicts_cached_etag_even_when_delete_fails(
    infra: DaprInfra, mock_load_with_etag: Mock
) -> None:
    """A failing store delete must still evict the cached etag (finally block)."""
    infra.get_state("boom")

    with patch.object(
        infra.state_store, "delete", side_effect=RuntimeError("store down")
    ):
        infra.purge_state("boom")

    assert _cache_key(infra, "boom") not in infra._etag_cache


def test_paired_get_then_save_drains_cached_etag(
    infra: DaprInfra, mock_load_with_etag: Mock
) -> None:
    """The optimization is preserved: a get_state followed by save_state for the
    same instance caches the etag on read and drains it on write."""
    saved: Dict[str, Any] = {}

    def fake_save(*, key: str, value: Any, **_: Any) -> None:
        saved[key] = value

    with patch.object(infra.state_store, "save", side_effect=fake_save):
        entry = infra.get_state("paired")
        assert _cache_key(infra, "paired") in infra._etag_cache

        infra.save_state("paired", entry=entry)
        assert _cache_key(infra, "paired") not in infra._etag_cache
