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

import time

from dapr_agents.observability.context_storage import WorkflowContextStorage


class TestWorkflowContextStorage:
    def test_store_and_get_roundtrip(self):
        storage = WorkflowContextStorage()
        storage.store_context("wf-1", {"traceparent": "abc"})

        assert storage.get_context("wf-1") == {"traceparent": "abc"}

    def test_get_missing_returns_none(self):
        storage = WorkflowContextStorage()

        assert storage.get_context("missing") is None

    def test_cleanup_removes_entry(self):
        storage = WorkflowContextStorage()
        storage.store_context("wf-1", {"traceparent": "abc"})

        storage.cleanup_context("wf-1")

        assert storage.get_context("wf-1") is None

    def test_cleanup_missing_entry_does_not_raise(self):
        storage = WorkflowContextStorage()
        storage.cleanup_context("does-not-exist")  # should not raise

    def test_entries_expire_after_ttl_without_explicit_cleanup(self):
        """Regression test: nothing in the wrappers ever calls
        cleanup_context() for a completed workflow (only store_context() is
        called, from wrappers/workflow.py), so entries must expire on their
        own or storage grows without bound for the life of the process."""
        storage = WorkflowContextStorage(ttl=0.05, maxsize=100)
        storage.store_context("wf-1", {"traceparent": "abc"})
        assert storage.get_context("wf-1") is not None

        time.sleep(0.1)

        assert storage.get_context("wf-1") is None
        assert storage.get_storage_stats()["stored_instances"] == 0

    def test_maxsize_bounds_storage_growth(self):
        storage = WorkflowContextStorage(maxsize=5, ttl=3600.0)

        for i in range(20):
            storage.store_context(f"wf-{i}", {"traceparent": str(i)})

        assert storage.get_storage_stats()["stored_instances"] <= 5

    def test_get_storage_stats_reports_instance_ids(self):
        storage = WorkflowContextStorage()
        storage.store_context("wf-1", {"traceparent": "abc"})

        stats = storage.get_storage_stats()

        assert stats["stored_instances"] == 1
        assert "wf-1" in stats["instance_ids"]
