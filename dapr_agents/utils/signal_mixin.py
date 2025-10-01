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

        # Check if we're in the main thread
        import threading

        if threading.current_thread() is not threading.main_thread():
            logger.debug("Skipping signal handler setup - not in main thread")
            return

        try:
            # Initialize shutdown event if it doesn't exist
            if not hasattr(self, "_shutdown_event") or self._shutdown_event is None:
                self._shutdown_event = asyncio.Event()

            # Try to get the current event loop, but don't create one if it doesn't exist
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, try to get the event loop for the current thread
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    logger.debug("No event loop available for signal handlers")
                    return

            # Set up signal handlers
            add_signal_handlers_cross_platform(loop, self._handle_shutdown_signal)

            self._signal_handlers_setup = True
            logger.debug("Signal handlers set up for graceful shutdown")
        except Exception as e:
            logger.debug(f"Could not set up signal handlers: {e}")
            # Don't fail initialization if signal handlers can't be set up

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
            try:
                # Call synchronously since we're in a signal handler
                import asyncio

                if asyncio.iscoroutinefunction(self.graceful_shutdown):
                    logger.debug("Async graceful shutdown - shutdown event set")
                else:
                    self.graceful_shutdown()  # type: ignore[unused-coroutine]
            except Exception as e:
                logger.debug(f"Error in graceful shutdown: {e}")

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
