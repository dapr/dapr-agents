import asyncio
import inspect
import logging
from typing import Optional

from dapr_agents.utils import add_signal_handlers_cross_platform

logger = logging.getLogger(__name__)


class ServiceMixin:
    """
    Mixin providing FastAPI service integration and lifecycle management for agentic workflows.

    Features:
        - Initializes and manages a FastAPI server for agent workflows.
        - Registers HTTP endpoints for workflow status, initiation, and custom user routes.
        - Supports both FastAPI service mode and headless (no HTTP server) operation.
        - Handles graceful shutdown via signal handling and resource cleanup.
        - Integrates workflow execution via HTTP POST and custom endpoints.
        - Manages subscription cleanup and workflow runtime shutdown on service stop.
        - Provides property access to the FastAPI app instance.
    """

    wf_runtime_is_running: Optional[bool] = None

    @property
    def app(self):
        """
        Return the FastAPI application initialized via ``as_service``.

        Returns:
            FastAPI: The FastAPI app instance.

        Raises:
            RuntimeError: If the FastAPI server has not been initialized.
        """
        if self._http_server:
            return self._http_server.app
        raise RuntimeError("FastAPI server not initialized. Call `as_service()` first.")

    def register_routes(self):
        """
        Register user-defined FastAPI routes decorated with ``@route``.
        """
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if getattr(method, "_is_fastapi_route", False):
                path = getattr(method, "_route_path")
                method_type = getattr(method, "_route_method", "GET")
                extra_kwargs = getattr(method, "_route_kwargs", {})
                logger.info(f"Registering route {method_type} {path} -> {name}")
                self.app.add_api_route(
                    path, method, methods=[method_type], **extra_kwargs
                )

    def as_service(self, port: Optional[int] = None, host: str = "0.0.0.0"):
        """
        Enable FastAPI service mode for the agent.

        Args:
            port: Required port number.
            host: Host address to bind to.

        Returns:
            self

        Raises:
            ValueError: If port is not provided.
        """
        from dapr_agents.service.fastapi import FastAPIServerBase

        if port is None:
            raise ValueError("Port must be provided as a parameter")

        self._http_server = FastAPIServerBase(
            service_name=self.name,
            service_port=port,
            service_host=host,
        )

        self.app.add_api_route("/status", lambda: {"ok": True})
        self.app.add_api_route(
            "/start-workflow", self.run_workflow_from_request, methods=["POST"]
        )

        self.register_routes()
        return self

    def handle_shutdown_signal(self, sig):
        """
        Handle shutdown signal and trigger graceful service stop.

        Args:
            sig: The received signal.
        """
        logger.info(f"Shutdown signal {sig} received. Stopping service gracefully...")
        self._shutdown_event.set()
        asyncio.create_task(self.stop())

    async def start(self):
        """
        Start the agent workflow service.

        This method starts the FastAPI server or runs in headless mode.
        """
        if self._is_running:
            logger.warning(
                "Service is already running. Ignoring duplicate start request."
            )
            return

        logger.info("Starting Agent Workflow Service...")
        self._shutdown_event.clear()

        try:
            if not hasattr(self, "_http_server") or self._http_server is None:
                logger.info("Running in headless mode.")
                loop = asyncio.get_event_loop()
                add_signal_handlers_cross_platform(loop, self.handle_shutdown_signal)
                self.register_message_routes()
                self._is_running = True
                while not self._shutdown_event.is_set():
                    await asyncio.sleep(1)
            else:
                logger.info("Running in FastAPI service mode.")
                self.register_message_routes()
                self._is_running = True
                await self._http_server.start()
        except asyncio.CancelledError:
            logger.info("Service received cancellation signal.")
        finally:
            await self.stop()

    async def stop(self):
        """
        Stop the agent workflow service and clean up resources.
        """
        if not self._is_running:
            logger.warning("Service is not running. Ignoring stop request.")
            return

        logger.info("Stopping Agent Workflow Service...")

        for (pubsub_name, topic_name), close_fn in self._subscriptions.items():
            try:
                logger.info(
                    f"Unsubscribing from pubsub '{pubsub_name}' topic '{topic_name}'"
                )
                close_fn()
            except Exception as e:
                logger.error(f"Failed to unsubscribe from topic '{topic_name}': {e}")

        self._subscriptions.clear()

        if hasattr(self, "_http_server") and self._http_server:
            logger.info("Stopping FastAPI server...")
            await self._http_server.stop()

        if getattr(self, "_wf_runtime_is_running", False):
            logger.info("Shutting down workflow runtime.")
            self.stop_runtime()
            self.wf_runtime_is_running = False

        self._is_running = False
        logger.info("Agent Workflow Service stopped successfully.")
