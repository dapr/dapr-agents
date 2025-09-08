"""
Reusable signal handling mixin for graceful shutdown across different service types.
"""
import asyncio
import logging
from typing import Optional

from dapr_agents.utils import add_signal_handlers_cross_platform

logger = logging.getLogger(__name__)


class SignalHandlingMixin:
    """
    Mixin providing reusable signal handling for graceful shutdown.

    This mixin can be used by any class that needs to handle shutdown signals
    (SIGINT, SIGTERM) gracefully. It provides a consistent interface for:
    - Setting up signal handlers
    - Managing shutdown events
    - Triggering graceful shutdown logic
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shutdown_event: Optional[asyncio.Event] = None
        self._signal_handlers_setup = False

    def setup_signal_handlers(self) -> None:
        """
        Set up signal handlers for graceful shutdown.

        This method should be called during initialization or startup
        to enable graceful shutdown handling.
        """
        # Initialize the attribute if it doesn't exist
        if not hasattr(self, "_signal_handlers_setup"):
            self._signal_handlers_setup = False

        if self._signal_handlers_setup:
            logger.debug("Signal handlers already set up")
            return

        # Initialize shutdown event if it doesn't exist
        if not hasattr(self, "_shutdown_event") or self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        add_signal_handlers_cross_platform(loop, self._handle_shutdown_signal)

        self._signal_handlers_setup = True
        logger.debug("Signal handlers set up for graceful shutdown")

    def _handle_shutdown_signal(self, sig: int) -> None:
        """
        Internal signal handler that triggers graceful shutdown.

        Args:
            sig: The received signal number
        """
        logger.debug(f"Shutdown signal {sig} received. Triggering graceful shutdown...")

        # Set the shutdown event
        if self._shutdown_event:
            self._shutdown_event.set()

        # Call the graceful shutdown method if it exists
        if hasattr(self, "graceful_shutdown"):
            asyncio.create_task(self.graceful_shutdown())
        elif hasattr(self, "stop"):
            # Fallback to stop() method if graceful_shutdown doesn't exist
            asyncio.create_task(self.stop())
        else:
            logger.warning(
                "No graceful shutdown method found. Implement graceful_shutdown() or stop() method."
            )

    async def graceful_shutdown(self) -> None:
        """
        Perform graceful shutdown operations.

        This method should be overridden by classes that use this mixin
        to implement their specific shutdown logic.

        Default implementation calls stop() if it exists.
        """
        if hasattr(self, "stop"):
            await self.stop()
        else:
            logger.warning(
                "No stop() method found. Override graceful_shutdown() to implement shutdown logic."
            )

    def is_shutdown_requested(self) -> bool:
        """
        Check if a shutdown has been requested.

        Returns:
            bool: True if shutdown has been requested, False otherwise
        """
        return (
            hasattr(self, "_shutdown_event")
            and self._shutdown_event is not None
            and self._shutdown_event.is_set()
        )

    async def wait_for_shutdown(self, check_interval: float = 1.0) -> None:
        """
        Wait for a shutdown signal to be received.

        Args:
            check_interval: How often to check for shutdown (in seconds)
        """
        if not hasattr(self, "_shutdown_event") or self._shutdown_event is None:
            raise RuntimeError(
                "Signal handlers not set up. Call setup_signal_handlers() first."
            )

        while not self._shutdown_event.is_set():
            await asyncio.sleep(check_interval)
