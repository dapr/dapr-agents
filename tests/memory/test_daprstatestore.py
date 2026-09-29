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

"""Unit tests for ConversationDaprStateMemory."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from dapr_agents.memory.daprstatestore import ConversationDaprStateMemory


def _make_memory() -> ConversationDaprStateMemory:
    """Build a memory instance whose Dapr store is a mock that captures writes."""
    memory = ConversationDaprStateMemory(agent_name="test-agent")
    store = MagicMock()
    store.get_state.return_value = MagicMock(data=None, etag=None)
    memory.dapr_store = store
    return memory


def test_add_message_records_created_at_in_utc(monkeypatch):
    """createdAt must be a timezone-aware UTC timestamp, not local time mislabeled
    as UTC.

    Regression for a bug where ``datetime.now().isoformat() + "Z"`` stamped the
    host's naive LOCAL time and hard-appended ``Z``, so on any non-UTC host the
    persisted instant was wrong by the host's UTC offset while claiming to be UTC.
    """
    # A local wall clock 4 hours ahead of UTC (e.g. a non-UTC deployment).
    true_utc = datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=timezone.utc)
    naive_local = (true_utc + timedelta(hours=4)).replace(tzinfo=None)

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            # Mirror the stdlib: no tz -> naive local time; tz -> that timezone.
            return naive_local if tz is None else true_utc.astimezone(tz)

    monkeypatch.setattr("dapr_agents.memory.daprstatestore.datetime", FakeDateTime)

    memory = _make_memory()
    memory.add_message({"role": "user", "content": "hello"}, "wf-1")

    saved_value = memory.dapr_store.save_state.call_args.args[1]
    stored = json.loads(saved_value)[0]
    parsed = datetime.fromisoformat(stored["createdAt"])

    # Must be timezone-aware and equal to the true UTC instant.
    assert parsed.tzinfo is not None
    assert parsed == true_utc
    # And must NOT be the naive local wall clock mislabeled as UTC.
    assert parsed != naive_local.replace(tzinfo=timezone.utc)


def test_get_messages_returns_most_recent_when_over_limit():
    """get_messages(limit=N) must return the N most recently added messages, not
    the N oldest.

    add_message always appends, so stored messages are ordered oldest to newest;
    slicing from the front returns the same first N messages forever once a
    conversation grows past the limit, never surfacing anything new.
    """
    memory = _make_memory()
    stored = [{"role": "user", "content": f"message-{i}"} for i in range(5)]
    memory.dapr_store.get_state.return_value = MagicMock(
        data=json.dumps(stored), etag="etag-1"
    )

    messages = memory.get_messages("wf-1", limit=2)

    assert [m["content"] for m in messages] == ["message-3", "message-4"]


def test_get_messages_with_limit_zero_returns_empty():
    """get_messages(limit=0) must return no messages.

    raw_messages[-0:] is raw_messages[0:], the entire list, since -0 == 0 in
    Python slicing. Without an explicit guard, limit=0 silently returns
    everything instead of nothing.
    """
    memory = _make_memory()
    stored = [{"role": "user", "content": f"message-{i}"} for i in range(5)]
    memory.dapr_store.get_state.return_value = MagicMock(
        data=json.dumps(stored), etag="etag-1"
    )

    messages = memory.get_messages("wf-1", limit=0)

    assert messages == []
