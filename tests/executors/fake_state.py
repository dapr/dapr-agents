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

"""In-memory stand-in for ``StateStoreService`` used by session store tests."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from dapr_agents.storage.daprstores.stateservice import StateStoreError


class FakeStateStore:
    """In-memory ``StateStoreService`` with ETags and first-write semantics."""

    def __init__(self) -> None:
        self.data: Dict[str, str] = {}
        self.versions: Dict[str, int] = {}
        self.conflicts_to_inject = 0
        self.fail_deletes = False
        self.saves: List[Dict[str, Any]] = []

    def load_with_etag(self, *, key, default=None, state_metadata=None):
        if key not in self.data:
            return default, None
        return json.loads(self.data[key]), str(self.versions[key])

    def save(
        self,
        *,
        key,
        value,
        etag=None,
        state_metadata=None,
        state_options=None,
        ttl_in_seconds=None,
    ):
        self.saves.append({"key": key, "etag": etag, "ttl": ttl_in_seconds})
        if state_options is not None:
            if etag is None and key in self.data:
                raise StateStoreError("first-write: key exists")
            if etag is not None and str(self.versions.get(key)) != etag:
                raise StateStoreError("etag mismatch")
            if self.conflicts_to_inject and "#" not in key:
                # Simulate a concurrent writer committing first.
                self.conflicts_to_inject -= 1
                self.versions[key] = self.versions.get(key, 0) + 1
                raise StateStoreError("etag mismatch")
        self.data[key] = json.dumps(value)
        self.versions[key] = self.versions.get(key, 0) + 1

    def load_many(self, keys, state_metadata=None):
        return {k: json.loads(self.data[k]) for k in keys if k in self.data}

    def delete(self, *, key, **_: Any):
        if self.fail_deletes and "#" in key:
            raise StateStoreError("delete failed")
        self.data.pop(key, None)
