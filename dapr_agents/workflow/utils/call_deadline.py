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

"""Run blocking calls with a deadline on a small pool of daemon threads.

``concurrent.futures.ThreadPoolExecutor`` joins its workers at interpreter
exit, so one hung sidecar call would block shutdown. This pool uses daemon
threads instead, grows only when every worker has a job, and is capped, so hung
calls cannot pile up threads without bound.

A call that times out keeps running on its worker until it returns; only the
caller stops waiting. A call still queued when its deadline passes never runs.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_WORKERS = 16

_Job = tuple["Future[Any]", Callable[..., Any], tuple[Any, ...], dict[str, Any]]


class DeadlineCallerClosedError(RuntimeError):
    """Raised when a call is submitted after :meth:`DeadlineCaller.close`."""


class DeadlineCaller:
    """Runs blocking callables on daemon worker threads and waits with a timeout."""

    def __init__(
        self,
        *,
        max_workers: int = DEFAULT_MAX_WORKERS,
        thread_name_prefix: str = "deadline-call",
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers!r}.")
        self._max_workers = max_workers
        self._prefix = thread_name_prefix
        self._queue: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._pending = 0  # submitted jobs not finished (queued or running)
        self._closed = False
        self._lock = threading.Lock()

    def call(
        self, fn: Callable[..., T], timeout: float, /, *args: Any, **kwargs: Any
    ) -> T:
        """Run ``fn(*args, **kwargs)`` on a worker and wait at most ``timeout`` seconds.

        Raises:
            TimeoutError: If the call did not finish in time (it may still
                complete in the background).
            DeadlineCallerClosedError: If the caller was closed.
            Exception: Whatever ``fn`` raised.
        """
        future: Future[T] = Future()
        self._submit((future, fn, args, kwargs))
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()  # a still-queued call is skipped
            raise

    def close(self) -> None:
        """Stop the workers once they finish their current call. Never blocks."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            workers = len(self._threads)
        for _ in range(workers):
            self._queue.put(None)

    def _submit(self, job: _Job) -> None:
        with self._lock:
            if self._closed:
                raise DeadlineCallerClosedError("DeadlineCaller is closed.")
            self._pending += 1
            workers = len(self._threads)
            if self._pending > workers and workers < self._max_workers:
                self._spawn()
            self._queue.put(job)

    def _spawn(self) -> None:
        thread = threading.Thread(
            target=self._work,
            name=f"{self._prefix}-{len(self._threads)}",
            daemon=True,
        )
        self._threads.append(thread)
        thread.start()

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            future, fn, args, kwargs = job
            if not future.set_running_or_notify_cancel():
                self._finish()  # cancelled while queued: skip it
                continue
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                self._finish()
                future.set_exception(exc)
            else:
                # Finish before the caller wakes, so its next call reuses this worker.
                self._finish()
                future.set_result(result)

    def _finish(self) -> None:
        with self._lock:
            self._pending -= 1


__all__ = ["DEFAULT_MAX_WORKERS", "DeadlineCaller", "DeadlineCallerClosedError"]
