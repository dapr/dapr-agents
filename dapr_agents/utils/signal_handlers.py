import asyncio
import signal
import platform
import logging
from typing import Callable, Any, Union

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
    if platform.system() == "Windows":
        # Windows uses traditional signal handlers
        for sig in signals:
            try:

                def windows_handler(s: int, f: Any) -> None:
                    asyncio.create_task(handler_func(s))

                signal.signal(sig, windows_handler)
            except Exception as e:
                logger.warning(f"Failed to register signal handler for {sig}: {e}")
    else:
        # Unix-like systems use asyncio signal handlers
        for sig in signals:
            try:

                def unix_handler() -> None:
                    asyncio.create_task(handler_func(sig))

                loop.add_signal_handler(sig, unix_handler)
            except NotImplementedError:
                logger.warning(f"Signal {sig} not supported in this environment")
            except Exception as e:
                logger.warning(f"Failed to register signal handler for {sig}: {e}")
