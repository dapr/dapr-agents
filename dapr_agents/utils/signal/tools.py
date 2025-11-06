from __future__ import annotations

import asyncio
import logging
import signal
import threading
from typing import Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

HandlerCleanup = Callable[[], None]


def _is_main_thread() -> bool:
    """
    Return True if running in the main thread.

    Signal handlers must be registered in the main thread; most runtimes
    will ignore or error otherwise.
    """
    return threading.current_thread() is threading.main_thread()


def install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    on_signal: Callable[[int], None],
    signals_to_handle: Iterable[signal.Signals] = (signal.SIGINT, signal.SIGTERM),
) -> HandlerCleanup:
    """
    Install Ctrl+C (SIGINT) / SIGTERM handlers in a cross-platform way.

    On Unix:
      - Prefer `loop.add_signal_handler` (non-blocking, runs in the loop thread).
      - If unsupported (e.g., loop not running), fall back to `signal.signal`
        and forward into the loop via `loop.call_soon_threadsafe`.

    On Windows:
      - Use `signal.signal`, forwarding into the loop with `loop.call_soon_threadsafe`.

    Handlers are only installed in the main thread.

    Args:
        loop: The asyncio event loop to schedule work onto after a signal arrives.
        on_signal: Callback invoked with the signal number (int). This is called
            in the loop thread (via `call_soon_threadsafe`) or directly as a
            best-effort fallback if the loop is already closed.
        signals_to_handle: Iterable of signals to install (default: SIGINT, SIGTERM).

    Returns:
        A zero-arg callable that cleans up the installed handlers. Always safe to call.

    Raises:
        ValueError: If `on_signal` is not callable.
    """
    if not callable(on_signal):
        raise ValueError("on_signal must be callable")

    if not _is_main_thread():
        logger.debug("Skipping signal handler installation (not in main thread).")
        return lambda: None

    sigs: List[signal.Signals] = list(signals_to_handle)
    previous: Dict[signal.Signals, object] = {}
    added_via_loop = False

    def _dispatch(sig_num: int) -> None:
        """Forward the signal to the loop thread safely."""
        try:
            loop.call_soon_threadsafe(on_signal, sig_num)
        except RuntimeError:
            # Loop likely closed; fall back to direct call (best effort).
            try:
                on_signal(sig_num)
            except Exception:
                logger.exception("Error dispatching shutdown after loop close")

    # Try loop-integrated handlers first (Unix). May raise on Windows or early init.
    try:
        for s in sigs:
            loop.add_signal_handler(s, _dispatch, s)
        added_via_loop = True
        logger.debug("Installed signal handlers via loop.add_signal_handler")
    except (NotImplementedError, RuntimeError):
        # Fallback: traditional handlers + bounce to loop
        for s in sigs:
            try:
                previous[s] = signal.getsignal(s)

                def _handler(signum: int, _frame) -> None:  # noqa: ANN001
                    _dispatch(signum)

                signal.signal(s, _handler)
            except Exception:
                logger.exception("Failed to install handler for %s", s)

        logger.debug("Installed signal handlers via signal.signal fallback")

    def _cleanup() -> None:
        """Remove loop handlers or restore previous traditional handlers."""
        if added_via_loop:
            for s in sigs:
                try:
                    loop.remove_signal_handler(s)
                except Exception:
                    # Loop may already be closed; ignore during shutdown.
                    pass
            return

        for s, prev in previous.items():
            try:
                # Only pass prev to signal.signal if it's a valid handler type
                if prev is None or callable(prev) or isinstance(prev, int):
                    signal.signal(s, prev)
                else:
                    signal.signal(s, None)
            except Exception:
                # Avoid raising from cleanup paths.
                pass

    return _cleanup


def uninstall_signal_handlers(cleanup: Optional[HandlerCleanup]) -> None:
    """
    Execute the cleanup closure returned by `install_signal_handlers`, if any.

    Args:
        cleanup: The cleanup function returned by `install_signal_handlers()`.

    Returns:
        None

    Notes:
        Safe to call multiple times and never raises.
    """
    if cleanup is None:
        return
    try:
        cleanup()
    except Exception:
        # Never raise during shutdown cleanup.
        logger.debug("Error while uninstalling signal handlers", exc_info=True)
