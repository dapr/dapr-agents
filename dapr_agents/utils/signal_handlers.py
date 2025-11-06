import asyncio
import signal
import platform
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


def add_signal_handlers_cross_platform(
    loop: asyncio.AbstractEventLoop,
    handler_func: Callable[[int], Any],
    signals=(signal.SIGINT, signal.SIGTERM),
):
    """
    Add signal handlers in a cross-platform way to allow for graceful shutdown.

    Because Windows/WSL2 signal handlers do not support asyncio,
    we support a cross platform means of handling signals for graceful shutdowns.

    Args:
        loop: The asyncio event loop
        handler_func: The function to call when signals are received
        signals: Tuple of signals to handle (default: SIGINT, SIGTERM)
    """
    import threading

    # Only set up signal handlers in the main thread
    if threading.current_thread() is not threading.main_thread():
        logger.debug("Skipping signal handler setup - not in main thread")
        return

    if platform.system() == "Windows":
        # Windows uses traditional signal handlers
        for sig in signals:
            try:

                def windows_handler(s: int, f: Any) -> None:
                    try:
                        handler_func(s)
                    except Exception as e:
                        logger.debug(f"Error in signal handler: {e}")

                signal.signal(sig, windows_handler)
            except Exception as e:
                logger.warning(f"Failed to register signal handler for {sig}: {e}")
    else:
        # Unix-like systems - use traditional signal handlers to avoid asyncio cleanup issues
        for sig in signals:
            try:

                def unix_handler(s: int, f: Any) -> None:
                    try:
                        handler_func(s)
                    except Exception as e:
                        logger.debug(f"Error in signal handler: {e}")

                # Use traditional signal handler instead of asyncio signal handler
                signal.signal(sig, unix_handler)
            except Exception as e:
                logger.warning(f"Failed to register signal handler for {sig}: {e}")
