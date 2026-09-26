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
threads instead and grows only when every worker has a job.

A call that times out keeps running on its worker until it returns; only the
caller stops waiting. A call still queued when its deadline passes never runs.

A call that never returns would hold its worker forever, so the pool bounds
the damage:

* A worker whose call has run longer than its deadline plus
  ``stuck_grace_seconds`` is presumed stuck (logged at WARNING) and no longer
  counts toward ``max_workers``, so a replacement can be started.
* ``max_threads`` caps the total number of threads, stuck ones included. When
  it is reached (logged at WARNING) no replacement is started.
* At most ``max_backlog`` calls may wait in the queue. Beyond that a call fails
  at once with :class:`DeadlineCallerBusyError` (a ``TimeoutError``) and never
  runs, instead of the queue growing without bound. A call that timed out
  while queued keeps its place until a worker skips it.

A presumed-stuck worker that returns after all keeps serving calls, unless the
pool already has ``max_workers`` healthy workers, in which case it exits.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_WORKERS = 16
# Default ``max_threads`` and ``max_backlog`` as multiples of ``max_workers``.
DEFAULT_THREAD_CEILING_FACTOR = 4
DEFAULT_BACKLOG_FACTOR = 4
DEFAULT_STUCK_GRACE_SECONDS = 5.0


class DeadlineCallerClosedError(RuntimeError):
    """Raised when a call is submitted after :meth:`DeadlineCaller.close`."""


class DeadlineCallerBusyError(TimeoutError):
    """Raised when the queue of waiting calls is full; the call never runs.

    A ``TimeoutError`` so callers treat it like a call that ran out of time.
    """


@dataclass(frozen=True)
class _Job:
    future: Future[Any]
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = None


@dataclass(frozen=True)
class _Running:
    started: float
    timeout: float | None


def _validated_limits(
    max_workers: int,
    max_threads: int | None,
    max_backlog: int | None,
    stuck_grace_seconds: float,
) -> tuple[int, int]:
    """Validate the pool limits; return (max_threads, max_backlog) with defaults."""
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers!r}.")
    ceiling = (
        max_workers * DEFAULT_THREAD_CEILING_FACTOR
        if max_threads is None
        else max_threads
    )
    if ceiling < max_workers:
        raise ValueError(
            f"max_threads must be >= max_workers ({max_workers}), got {ceiling!r}."
        )
    backlog = (
        max_workers * DEFAULT_BACKLOG_FACTOR if max_backlog is None else max_backlog
    )
    if backlog < 1:
        raise ValueError(f"max_backlog must be >= 1, got {backlog!r}.")
    if stuck_grace_seconds < 0:
        raise ValueError(
            f"stuck_grace_seconds must be >= 0, got {stuck_grace_seconds!r}."
        )
    return ceiling, backlog


class DeadlineCaller:
    """Runs blocking callables on daemon worker threads and waits with a timeout."""

    def __init__(
        self,
        *,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_threads: int | None = None,
        max_backlog: int | None = None,
        stuck_grace_seconds: float = DEFAULT_STUCK_GRACE_SECONDS,
        thread_name_prefix: str = "deadline-call",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            max_workers: Healthy workers the pool grows to.
            max_threads: Hard ceiling on threads, presumed-stuck ones included.
                Default ``4 * max_workers``.
            max_backlog: Most calls that may wait for a worker. Default
                ``4 * max_workers``.
            stuck_grace_seconds: How long past its deadline a call may run
                before its worker is presumed stuck.
            thread_name_prefix: Worker thread name prefix.
            clock: Monotonic clock (injectable for tests).
        """
        ceiling, backlog = _validated_limits(
            max_workers, max_threads, max_backlog, stuck_grace_seconds
        )
        self._max_workers = max_workers
        self._max_threads = ceiling
        self._max_backlog = backlog
        self._grace = stuck_grace_seconds
        self._prefix = thread_name_prefix
        self._clock = clock
        self._queue: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._threads: dict[int, threading.Thread] = {}
        self._running: dict[int, _Running] = {}
        self._reported_stuck: set[int] = set()
        self._queued = 0  # submitted jobs no worker has taken yet
        self._ids = itertools.count()
        self._ceiling_reported = False
        self._backlog_reported = False
        self._closed = False
        self._lock = threading.Lock()

    def call(
        self, fn: Callable[..., T], timeout: float, /, *args: Any, **kwargs: Any
    ) -> T:
        """Run ``fn(*args, **kwargs)`` on a worker and wait at most ``timeout`` seconds.

        Raises:
            TimeoutError: If the call did not finish in time (it may still
                complete in the background).
            DeadlineCallerBusyError: If the backlog is full (the call never runs).
            DeadlineCallerClosedError: If the caller was closed.
            Exception: Whatever ``fn`` raised.
        """
        future = self.start(fn, timeout, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()  # a still-queued call is skipped
            raise

    def start(
        self, fn: Callable[..., T], timeout: float, /, *args: Any, **kwargs: Any
    ) -> Future[T]:
        """Queue ``fn(*args, **kwargs)`` and return its future.

        ``timeout`` is the deadline the caller will wait; a call that runs past
        it plus the grace period marks its worker as presumed stuck. The caller
        waits on the future itself; cancelling it before a worker picks it up
        skips the call.

        Raises:
            DeadlineCallerBusyError: If the backlog is full (the call never runs).
            DeadlineCallerClosedError: If the caller was closed.
        """
        future: Future[T] = Future()
        self._enqueue(_Job(future, fn, args, kwargs, timeout))
        return future

    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        """Like :meth:`start` without a deadline: the worker is never presumed stuck."""
        future: Future[T] = Future()
        self._enqueue(_Job(future, fn, args, kwargs))
        return future

    def close(self) -> None:
        """Stop the workers once they finish their current call. Never blocks."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            workers = len(self._threads)
        for _ in range(workers):
            self._queue.put(None)

    # ---- internals -------------------------------------------------------------

    def _enqueue(self, job: _Job) -> None:
        with self._lock:
            if self._closed:
                raise DeadlineCallerClosedError("DeadlineCaller is closed.")
            if self._queued >= self._max_backlog:
                first = not self._backlog_reported
                self._backlog_reported = True
                self._raise_busy(first)
            self._queued += 1
            self._grow_if_needed()
            self._queue.put(job)

    def _raise_busy(self, log: bool) -> None:
        if log:
            logger.warning(
                "%s pool: %d calls already wait for a worker (limit); failing new "
                "calls at once until the backlog drains.",
                self._prefix,
                self._queued,
            )
        raise DeadlineCallerBusyError(
            f"{self._prefix} pool backlog is full ({self._max_backlog} calls)."
        )

    def _grow_if_needed(self) -> None:
        """Start a worker when queued jobs outnumber idle workers (lock held)."""
        idle = len(self._threads) - len(self._running)
        if self._queued <= idle:
            return
        healthy = len(self._threads) - self._count_stuck()
        if healthy >= self._max_workers:
            return
        if len(self._threads) >= self._max_threads:
            if not self._ceiling_reported:
                self._ceiling_reported = True
                logger.warning(
                    "%s pool: %d threads, the ceiling, with %d presumed stuck; not "
                    "starting a replacement. Calls wait or fail until one returns.",
                    self._prefix,
                    len(self._threads),
                    len(self._threads) - healthy,
                )
            return
        self._spawn()

    def _count_stuck(self) -> int:
        """Workers presumed stuck; each is logged once when first noticed (lock held)."""
        now = self._clock()
        stuck = 0
        for worker_id, running in self._running.items():
            if running.timeout is None:
                continue
            overdue = now - running.started - running.timeout
            if overdue <= self._grace:
                continue
            stuck += 1
            if worker_id not in self._reported_stuck:
                self._reported_stuck.add(worker_id)
                logger.warning(
                    "%s pool: a call has run %.1fs past its %gs deadline; presuming "
                    "its worker stuck and allowing a replacement.",
                    self._prefix,
                    overdue,
                    running.timeout,
                )
        return stuck

    def _spawn(self) -> None:
        worker_id = next(self._ids)
        thread = threading.Thread(
            target=self._work,
            args=(worker_id,),
            name=f"{self._prefix}-{worker_id}",
            daemon=True,
        )
        self._threads[worker_id] = thread
        thread.start()

    def _work(self, worker_id: int) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._retire(worker_id)
                return
            if not self._take(worker_id, job):
                continue  # cancelled while queued: skip it
            try:
                result = job.fn(*job.args, **job.kwargs)
            except BaseException as exc:
                retire = self._finish(worker_id)
                job.future.set_exception(exc)
            else:
                # Finish before the caller wakes, so its next call reuses this worker.
                retire = self._finish(worker_id)
                job.future.set_result(result)
            if retire:
                return

    def _take(self, worker_id: int, job: _Job) -> bool:
        """Dequeue bookkeeping; False when the job was cancelled while queued."""
        with self._lock:
            self._queued -= 1
            self._backlog_reported = False  # the backlog has room again
            if not job.future.set_running_or_notify_cancel():
                return False
            self._running[worker_id] = _Running(self._clock(), job.timeout)
            return True

    def _finish(self, worker_id: int) -> bool:
        """Mark the worker idle; True when it is surplus and should exit."""
        with self._lock:
            self._running.pop(worker_id, None)
            self._reported_stuck.discard(worker_id)
            healthy = len(self._threads) - self._count_stuck()
            if healthy <= self._max_workers:
                return False
            self._threads.pop(worker_id, None)
            self._ceiling_reported = False
            return True

    def _retire(self, worker_id: int) -> None:
        with self._lock:
            self._threads.pop(worker_id, None)


__all__ = [
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_STUCK_GRACE_SECONDS",
    "DeadlineCaller",
    "DeadlineCallerBusyError",
    "DeadlineCallerClosedError",
]
