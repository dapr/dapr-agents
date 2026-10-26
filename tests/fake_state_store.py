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

"""Shared in-memory stand-in for ``StateStoreService`` with ETag semantics."""

from __future__ import annotations

import copy
import json
import threading
from typing import Any, Dict, Iterable, List, Optional

from dapr_agents.storage.daprstores.stateservice import StateStoreError


class FakeEtagStateStore:
    """
    In-memory ``StateStoreService`` modelling Dapr optimistic concurrency.

    A save that passes ``state_options`` (first-write concurrency) must carry
    the current ETag, or no ETag when the key does not exist yet; otherwise it
    raises ``StateStoreError`` like a real conflict. Saves without options
    overwrite unconditionally. Values are stored as JSON, so callers never
    share mutable references with the store, and every operation holds a lock
    so concurrent writers can be modelled.

    Fault injection for callers that need it:

    * ``conflicts_to_inject``: the next N option-carrying saves to a key
      without ``#`` fail as if a concurrent writer committed first.
    * ``lost_replies_to_inject``: the next N option-carrying saves to a key
      without ``#`` are applied and then raise, as if the reply was lost.
    * ``fail_deletes``: deletes of keys containing ``#`` raise.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.data: Dict[str, str] = {}
        self.versions: Dict[str, int] = {}
        self.conflicts_to_inject = 0
        self.lost_replies_to_inject = 0
        self.fail_deletes = False
        self.saves: List[Dict[str, Any]] = []
        self.deleted: List[str] = []

    def seed(self, key: str, value: Any) -> None:
        with self._lock:
            self._store(key, value)

    def load_with_etag(self, *, key, default=None, state_metadata=None):
        with self._lock:
            if key not in self.data:
                return copy.deepcopy(default), None
            return json.loads(self.data[key]), str(self.versions[key])

    def load_many(self, keys: Iterable[str], state_metadata=None):
        with self._lock:
            return {k: json.loads(self.data[k]) for k in keys if k in self.data}

    def save(
        self,
        *,
        key,
        value,
        etag: Optional[str] = None,
        state_metadata=None,
        state_options=None,
        ttl_in_seconds: Optional[int] = None,
    ) -> None:
        with self._lock:
            self.saves.append({"key": key, "etag": etag, "ttl": ttl_in_seconds})
            if state_options is not None:
                self._check_concurrency(key, etag)
            self._store(key, value)
            if state_options is not None and self._take("lost_replies_to_inject", key):
                raise StateStoreError("deadline exceeded")

    def delete(self, *, key, **_: Any) -> None:
        with self._lock:
            if self.fail_deletes and "#" in key:
                raise StateStoreError("delete failed")
            self.deleted.append(key)
            self.data.pop(key, None)
            self.versions.pop(key, None)

    def _check_concurrency(self, key: str, etag: Optional[str]) -> None:
        current = self.versions.get(key)
        if etag is None and key in self.data:
            raise StateStoreError(f"first-write: key {key} exists")
        if etag is not None and str(current) != etag:
            raise StateStoreError(f"etag mismatch for {key}")
        if self._take("conflicts_to_inject", key):
            # Simulate a concurrent writer committing first.
            self.versions[key] = (current or 0) + 1
            raise StateStoreError(f"etag mismatch for {key}")

    def _take(self, counter: str, key: str) -> bool:
        if "#" in key or getattr(self, counter) <= 0:
            return False
        setattr(self, counter, getattr(self, counter) - 1)
        return True

    def _store(self, key: str, value: Any) -> None:
        self.data[key] = json.dumps(value)
        self.versions[key] = self.versions.get(key, 0) + 1
