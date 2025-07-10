from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import Field, ConfigDict
from typing import List, Optional, Any
from dapr_agents.service import APIServerBase
from dapr_agents.utils import add_signal_handlers_cross_platform
import uvicorn
import asyncio
import signal
import logging

logger = logging.getLogger(__name__)


class FastAPIServerBase(APIServerBase):
    """
    Abstract base class for FastAPI-based API server services.
    Provides core FastAPI functionality, with support for CORS, lifecycle management, and graceful shutdown.
    """

    description: Optional[str] = Field(
        None, description="Description of the API service."
    )
    cors_origins: Optional[List[str]] = Field(
        default_factory=lambda: ["*"], description="Allowed CORS origins."
    )
    cors_credentials: bool = Field(
        True, description="Whether to allow credentials in CORS requests."
    )
    cors_methods: Optional[List[str]] = Field(
        default_factory=lambda: ["*"], description="Allowed HTTP methods for CORS."
    )
    cors_headers: Optional[List[str]] = Field(
        default_factory=lambda: ["*"], description="Allowed HTTP headers for CORS."
    )

    # Fields initialized in model_post_init
    app: Optional[FastAPI] = Field(
        default=None, init=False, description="The FastAPI application instance."
    )
    server: Optional[Any] = Field(
        default=None,
        init=False,
        description="Server handle for running the FastAPI app.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to configure core FastAPI app and CORS settings.
        """

        # Initialize FastAPI app with title and description
        self.app = FastAPI(
            title=f"{self.service_name} API Server",
            description=self.description or self.service_name,
            lifespan=self.lifespan,
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=self.cors_credentials,
            allow_methods=self.cors_methods,
            allow_headers=self.cors_headers,
        )

        logger.info(
            f"{self.service_name} FastAPI server initialized on port {self.service_port} with CORS settings."
        )

        # Call the base post-initialization
        super().model_post_init(__context)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """
        Default lifespan function to manage startup and shutdown processes.
        Can be overridden by subclasses to add setup and teardown tasks such as handling agent metadata.
        """
        try:
            yield
        finally:
            await self.stop()

    async def start(self, log_level=None):
        """
        Start the FastAPI app server using the existing event loop with a specified logging level,
        and ensure that shutdown is handled gracefully with SIGINT and SIGTERM signals.
        """
        if log_level is None:
            log_level = logging.getLevelName(logger.getEffectiveLevel()).lower()

        # Set port to 0 if we want a random port
        requested_port = self.service_port or 0

        config = uvicorn.Config(
            self.app,
            host=self.service_host,
            port=requested_port,
            log_level=log_level,
        )
        self.server: uvicorn.Server = uvicorn.Server(config)

        # Add signal handlers using cross-platform approach for graceful shutdown
        loop = asyncio.get_event_loop()
        add_signal_handlers_cross_platform(loop, self.stop)

        # Start in background so we can inspect the actual port
        server_task = asyncio.create_task(self.server.serve())

        # Wait for startup to complete
        while not self.server.started:
            await asyncio.sleep(0.1)

        # Extract the real port from the bound socket
        if self.server.servers:
            sock = list(self.server.servers)[0].sockets[0]
            actual_port = sock.getsockname()[1]
            self.service_port = actual_port
        else:
            logger.warning(f"{self.service_name} could not determine bound port")

        await server_task

    async def stop(self):
        """
        Stop the FastAPI server gracefully.
        """
        if self.server:
            logger.info(
                f"Stopping {self.service_name} server on port {self.service_port}."
            )
            self.server.should_exit = True
