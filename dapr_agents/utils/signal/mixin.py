from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from .tools import install_signal_handlers, uninstall_signal_handlers

logger = logging.getLogger(__name__)


class SignalMixin:
    """
    Mixin providing graceful shutdown via Ctrl+C (SIGINT) / SIGTERM.

    Features:
      - `install_signal_handlers()` / `remove_signal_handlers()` to manage OS handlers.
      - An internal `asyncio.Event` waited by `wait_for_shutdown()`.
      - `request_shutdown()` to trigger the same path programmatically (thread-safe).
      - Overridable async `graceful_shutdown()` hook.
      - Schedules cleanup (never awaits) from the actual signal context.
      - Safe to reuse the same instance across starts/stops in tests.

    Typical usage::

        class Service(SignalMixin):
            async def start(self):
                self.install_signal_handlers()
                # ... start tasks/resources ...
                await self.wait_for_shutdown()
                await self.graceful_shutdown()
                self.remove_signal_handlers()

            async def graceful_shutdown(self):
                # ... stop tasks/resources, close clients ...
                ...
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize mixin state.

        Supports multiple inheritance (delegates to super()).
        """
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        self._shutdown_event: Optional[asyncio.Event] = None
        self._cleanup_handlers: Optional[Callable[[], None]] = None
        self._shutdown_task_scheduled: bool = False
        self._signal_loop: Optional[
            asyncio.AbstractEventLoop
        ] = None  # loop used for handler install
        self._last_signal: Optional[
            int
        ] = None  # last observed signal number (-1 if programmatic)

    # ------------------- Public API -------------------

    def install_signal_handlers(self) -> None:
        """
        Install OS signal handlers and initialize (or reinitialize) the shutdown event.

        Idempotent: safe to call multiple times.

        Also resets internal scheduling flags so the instance can be restarted.

        Returns:
            None

        Raises:
            RuntimeError: If no event loop can be obtained for the current thread.
        """
        # (Re)create a fresh event if first install OR previous event is already set.
        if self._shutdown_event is None or self._shutdown_event.is_set():
            self._shutdown_event = asyncio.Event()

        # Reset scheduling gate so new shutdowns can be scheduled after restart.
        self._shutdown_task_scheduled = False

        # If handlers are already installed and we have a loop, nothing to do.
        if self._cleanup_handlers is not None and self._signal_loop is not None:
            return

        # Capture the loop we will always bounce into (thread-safe).
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError as exc:
                raise RuntimeError(
                    "No asyncio event loop available to install signal handlers."
                ) from exc

        self._signal_loop = loop
        self._cleanup_handlers = install_signal_handlers(loop, self._on_signal)

    def remove_signal_handlers(self) -> None:
        """
        Remove/uninstall previously installed signal handlers.

        Safe to call multiple times; no-op if nothing is installed.

        Returns:
            None
        """
        uninstall_signal_handlers(self._cleanup_handlers)
        self._cleanup_handlers = None
        # Keep _signal_loop for request_shutdown() bounce to succeed even after uninstall.
        # You can clear it when disposing the object if desired.

    async def wait_for_shutdown(self) -> None:
        """
        Block until a shutdown is requested (signal or programmatic).

        Returns:
            None

        Raises:
            RuntimeError: If handlers/event have not been installed first.
        """
        if self._shutdown_event is None:
            raise RuntimeError(
                "Call install_signal_handlers() before wait_for_shutdown()."
            )
        await self._shutdown_event.wait()

    def is_shutdown_requested(self) -> bool:
        """
        Indicate whether shutdown has been requested.

        Returns:
            bool: True if a shutdown signal has been received or requested programmatically.
        """
        return bool(self._shutdown_event and self._shutdown_event.is_set())

    def request_shutdown(self) -> None:
        """
        Programmatically request shutdown (thread-safe, same path as OS signal).

        If a loop was captured during installation, this always uses
        `loop.call_soon_threadsafe(...)` so callers from foreign threads
        behave like real signals.

        Returns:
            None
        """
        if self._signal_loop is not None:
            try:
                self._signal_loop.call_soon_threadsafe(self._on_signal, -1)
                return
            except Exception:
                # Fall through to direct call if loop is already closed.
                pass
        self._on_signal(sig=-1)

    @property
    def last_signal(self) -> Optional[int]:
        """
        The last OS signal observed (or -1 if programmatic), else None.

        Returns:
            Optional[int]: Last observed signal.
        """
        return self._last_signal

    # ------------------- Hooks -------------------

    async def graceful_shutdown(self) -> None:
        """
        Override to perform async cleanup (close clients, cancel tasks, flush logs).

        Default implementation:
            If the class provides a `stop()` attribute callable, it is invoked.
            If `stop()` returns a coroutine, it is awaited.

        Returns:
            None
        """
        stop = getattr(self, "stop", None)
        if callable(stop):
            maybe_coro = stop()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro  # type: ignore[func-returns-value]

    # ------------------- Internals -------------------

    def _schedule_graceful_shutdown(self) -> None:
        """
        Schedule `graceful_shutdown()` on the captured loop exactly once,
        and reset the scheduling gate when it completes.

        This function is resilient to a stopped/closed loop:
        - If `call_soon_threadsafe` fails, the scheduling flag is reset.
        - We then try a best-effort fallback on the currently running loop (if any).
        """
        loop = self._signal_loop
        if loop is None:
            # Best-effort: try current running loop, then thread's loop.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    logger.debug(
                        "No event loop available to schedule graceful shutdown."
                    )
                    return

        def _mark_done(task: asyncio.Task) -> None:
            """
            Done-callback for the graceful shutdown task.

            - Resets the scheduling gate to allow reuse of the instance.
            - Surfaces exceptions from the shutdown task into logs.
            - Calls optional `on_shutdown_complete()` hook (non-async, best-effort).
            """
            try:
                exc = task.exception()
                if exc is not None:
                    logger.exception("graceful_shutdown() raised", exc_info=exc)
                hook = getattr(self, "on_shutdown_complete", None)
                if callable(hook):
                    try:
                        hook()
                    except Exception:
                        logger.debug("on_shutdown_complete hook raised", exc_info=True)
            except asyncio.CancelledError:
                logger.debug("graceful_shutdown() task was cancelled", exc_info=True)
            finally:
                # Allow reuse of the instance in tests / restarts.
                self._shutdown_task_scheduled = False

        # Try to bounce into the captured loop thread.
        try:

            def _spawn() -> None:
                t = loop.create_task(self.graceful_shutdown())
                t.add_done_callback(_mark_done)

            loop.call_soon_threadsafe(_spawn)
            return
        except Exception:
            # If scheduling on the captured loop failed, reset the gate so callers can retry.
            logger.debug(
                "Failed to schedule via call_soon_threadsafe; attempting fallback.",
                exc_info=True,
            )
            self._shutdown_task_scheduled = False

        # Fallback: try the *current* running loop (e.g., called from a different alive loop).
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop; nothing else we can do. Leave the gate open for future attempts.
            return

        try:
            # Re-set the gate since we're attempting a new schedule path.
            self._shutdown_task_scheduled = True
            task = current.create_task(self.graceful_shutdown())
            task.add_done_callback(_mark_done)
        except Exception:
            # Scheduling failed again; open the gate so a later attempt can retry.
            logger.debug("Fallback scheduling on current loop failed.", exc_info=True)
            self._shutdown_task_scheduled = False

    def _on_signal(self, sig: int) -> None:
        """
        Internal signal callback. Do not await here.

        Sets the shutdown event and schedules `graceful_shutdown()` (once).
        Records the signal and calls an optional non-blocking hook.

        Args:
            sig: The received signal number (or -1 if triggered programmatically).

        Returns:
            None
        """
        if self._shutdown_event is None:
            return

        # Record & optional hook for observability
        self._last_signal = sig
        hook = getattr(self, "on_signal_received", None)
        if callable(hook):
            try:
                hook(sig)  # tiny, non-blocking hook; do not await here
            except Exception:
                logger.debug("on_signal_received hook raised", exc_info=True)

        if not self._shutdown_event.is_set():
            self._shutdown_event.set()

        if self._shutdown_task_scheduled:
            return
        self._shutdown_task_scheduled = True

        logger.debug(
            "Shutdown requested (signal=%s); scheduling graceful shutdown.", sig
        )
        self._schedule_graceful_shutdown()
