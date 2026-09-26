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

"""Per-topic, in-process state a workflow event route keeps between deliveries.

* :class:`AttemptTracker` counts deliveries per message that ended without a
  decision: "instance not found", or a hook that did not answer.
* :class:`TimedOutRaises` remembers raises that timed out but kept running, so
  a redelivery of the same message does not raise the event a second time.

Both live in process memory only; a restart forgets them.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from cachetools import LRUCache

logger = logging.getLogger(__name__)

# Messages whose attempt count one tracker keeps.
NOT_FOUND_TRACKER_MAXSIZE = 4096
# Lower bound on how many finished timed-out raises one topic remembers.
MIN_TIMED_OUT_RAISES_TRACKED = 4096
# Hard cap on running timed-out raises one topic tracks. Each one also holds
# a sidecar-pool thread, so the thread ceiling is reached long before this.
MAX_RUNNING_RAISES_TRACKED = 16_384


class AttemptTracker:
    """Per-route, per-process count of undecided deliveries of each message.

    Used for "instance not found" deliveries and for hooks that timed out or
    could not run.

    Entries never expire by time, so a message redelivered slower than any
    TTL keeps its count; they are removed when the message is raised or given
    up. Bounded to ``maxsize`` messages: if more distinct messages than that
    are waiting for missing instances at once, the least recently used counts
    are evicted and those messages start counting again, so ``max_attempts`` /
    ``window_seconds`` are only guaranteed below that bound. Any publisher on
    the topic can trigger this by sending messages for ids that never exist.
    """

    def __init__(self, maxsize: int | None = None) -> None:
        size = NOT_FOUND_TRACKER_MAXSIZE if maxsize is None else maxsize
        self._cache: LRUCache = LRUCache(maxsize=size)
        self._lock = threading.Lock()

    def record(self, key: str, now: float) -> tuple[int, float]:
        """Count one more not-found delivery; return (attempts, first_seen)."""
        with self._lock:
            previous: tuple[int, float] | None = self._cache.get(key)
            updated = (1, now) if previous is None else (previous[0] + 1, previous[1])
            self._cache[key] = updated
            return updated

    def clear(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)


@dataclass(frozen=True)
class RaiseIdentity:
    """What a raise delivered: target instance, event name and payload digest."""

    instance_id: str
    event_name: str
    data_sha256: str


@dataclass(frozen=True)
class TrackedRaise:
    """A timed-out raise and the identity of the message that started it."""

    identity: RaiseIdentity
    future: Future[Any]


class TimedOutRaises:
    """Raises that timed out while still running, keyed by the message's dedupe key.

    A raise that is still running sits in a dict that capacity pressure never
    evicts, so a redelivery always finds a raise that may still land. That
    dict is capped at ``max_running``; beyond it a new raise is not tracked
    (:meth:`add` returns False). Once a raise finishes it moves to an LRU of
    ``maxsize`` entries, where it waits for the redelivery that reads it and
    then removes it.
    """

    def __init__(
        self, maxsize: int, max_running: int = MAX_RUNNING_RAISES_TRACKED
    ) -> None:
        self._running: dict[str, TrackedRaise] = {}
        self._settled: LRUCache = LRUCache(maxsize=maxsize)
        self._max_running = max_running
        self._lock = threading.Lock()

    def add(self, key: str, tracked: TrackedRaise) -> bool:
        """Track a timed-out raise; False when it could not be tracked.

        It is not tracked when ``max_running`` raises are already running, or
        when ``key`` belongs to a different raise that is still running.
        """
        with self._lock:
            current = self._running.get(key)
            if current is not None and current.identity != tracked.identity:
                return False
            if current is None and len(self._running) >= self._max_running:
                return False
            self._settled.pop(key, None)
            self._running[key] = tracked
        tracked.future.add_done_callback(lambda _: self._settle(key, tracked))
        return True

    def peek(self, key: str) -> TrackedRaise | None:
        """The raise tracked for ``key``, running or finished; nothing is removed."""
        with self._lock:
            tracked = self._running.get(key)
            if tracked is None:
                tracked = self._settled.get(key)
            return tracked

    def discard(self, key: str, tracked: TrackedRaise) -> None:
        """Forget ``tracked`` once a redelivery has used its outcome."""
        with self._lock:
            if self._running.get(key) is tracked:
                del self._running[key]
            if self._settled.get(key) is tracked:
                del self._settled[key]

    def running_count(self) -> int:
        with self._lock:
            return len(self._running)

    def __len__(self) -> int:
        with self._lock:
            return len(self._running) + len(self._settled)

    def _settle(self, key: str, tracked: TrackedRaise) -> None:
        with self._lock:
            if self._running.get(key) is tracked:
                del self._running[key]
                self._settled[key] = tracked


@dataclass(frozen=True)
class EventRouteTopicState:
    """The in-process state of one event-route topic."""

    not_found: AttemptTracker
    undecided: AttemptTracker
    timed_out: TimedOutRaises

    @classmethod
    def create(cls, dedupe_max_entries: int) -> EventRouteTopicState:
        """State sized from the route's ``dedupe_max_entries``."""
        return cls(
            not_found=AttemptTracker(),
            undecided=AttemptTracker(),
            timed_out=TimedOutRaises(
                max(dedupe_max_entries, MIN_TIMED_OUT_RAISES_TRACKED)
            ),
        )


__all__ = [
    "MAX_RUNNING_RAISES_TRACKED",
    "MIN_TIMED_OUT_RAISES_TRACKED",
    "NOT_FOUND_TRACKER_MAXSIZE",
    "EventRouteTopicState",
    "AttemptTracker",
    "RaiseIdentity",
    "TimedOutRaises",
    "TrackedRaise",
]
