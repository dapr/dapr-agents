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

"""Helpers for the calls a workflow event route makes.

* Classifying sidecar gRPC errors as permanent (never retried).
* Remembering raises that timed out but kept running, so a redelivery of the
  same message does not raise the event a second time.
* Running user hooks (``authorize`` and the event-route filters) with a
  deadline and a strict ``True`` contract.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from typing import Any, Callable

import grpc
from cachetools import LRUCache

from dapr_agents.workflow.utils.call_deadline import DeadlineCaller

logger = logging.getLogger(__name__)

# gRPC codes that will not change on a redelivery of the same message. They
# are handled like a terminal workflow (dead-letter or drop), never retried.
PERMANENT_SIDECAR_ERROR_CODES: frozenset[grpc.StatusCode] = frozenset(
    {
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.PERMISSION_DENIED,
        grpc.StatusCode.UNAUTHENTICATED,
        grpc.StatusCode.UNIMPLEMENTED,
        grpc.StatusCode.OUT_OF_RANGE,
        grpc.StatusCode.FAILED_PRECONDITION,
    }
)

# Lower bound on how many timed-out raises one topic remembers.
MIN_TIMED_OUT_RAISES_TRACKED = 4096


def permanent_error_code(exc: BaseException) -> grpc.StatusCode | None:
    """The gRPC status code of ``exc`` when it is a permanent sidecar error, else None."""
    if not isinstance(exc, grpc.RpcError):
        return None
    try:
        code = exc.code()  # type: ignore[attr-defined]
    except Exception:
        return None
    return code if code in PERMANENT_SIDECAR_ERROR_CODES else None


class TimedOutRaises:
    """Raises that timed out while still running, keyed by the message's dedupe key.

    Process memory only: a restart forgets them, so a redelivery after a
    restart can raise the event again. Size-bounded (least recently used
    entries are evicted); an entry is removed when a redelivery resolves it.
    """

    def __init__(self, maxsize: int) -> None:
        self._futures: LRUCache = LRUCache(maxsize=maxsize)
        self._lock = threading.Lock()

    def add(self, key: str, future: Future[Any]) -> None:
        with self._lock:
            self._futures[key] = future

    def get(self, key: str) -> Future[Any] | None:
        """The tracked future for ``key``; a finished one is removed as it is returned."""
        with self._lock:
            future: Future[Any] | None = self._futures.get(key)
            if future is not None and future.done():
                del self._futures[key]
            return future

    def __len__(self) -> int:
        with self._lock:
            return len(self._futures)


def run_strict_hook(
    caller: DeadlineCaller,
    hook: Callable[[Any, Any], Any],
    timeout: float,
    value: Any,
    msg_ctx: Any,
    *,
    kind: str,
    route_name: str,
) -> bool:
    """Run ``hook(value, msg_ctx)`` with a deadline; True only for an exact ``True``.

    A timeout, an exception or any other return value (``"False"``, ``1``,
    a mock) is a rejection. The value is never logged.
    """
    try:
        result = caller.call(hook, timeout, value, msg_ctx)
    except TimeoutError:
        logger.warning(
            "Event route %r: %s timed out after %gs; rejecting the message.",
            route_name,
            kind,
            timeout,
        )
        return False
    except Exception as exc:
        logger.warning(
            "Event route %r: %s raised %s; rejecting the message.",
            route_name,
            kind,
            type(exc).__name__,
        )
        return False
    if result is True:
        return True
    if result is not False:
        logger.warning(
            "Event route %r: %s returned %s, not a bool; rejecting the message.",
            route_name,
            kind,
            type(result).__name__,
        )
    return False


__all__ = [
    "MIN_TIMED_OUT_RAISES_TRACKED",
    "PERMANENT_SIDECAR_ERROR_CODES",
    "TimedOutRaises",
    "permanent_error_code",
    "run_strict_hook",
]
