from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import Field, ConfigDict
from typing import List, Optional, Any
from dapr_agents.service import APIServerBase
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

    description: Optional[str] = Field(None, description="Description of the API service.")
    cors_origins: Optional[List[str]] = Field(default_factory=lambda: ["*"], description="Allowed CORS origins.")
    cors_credentials: bool = Field(True, description="Whether to allow credentials in CORS requests.")
    cors_methods: Optional[List[str]] = Field(default_factory=lambda: ["*"], description="Allowed HTTP methods for CORS.")
    cors_headers: Optional[List[str]] = Field(default_factory=lambda: ["*"], description="Allowed HTTP headers for CORS.")

    # Fields initialized in model_post_init
    app: Optional[FastAPI] = Field(default=None, init=False, description="The FastAPI application instance.")
    server: Optional[Any] = Field(default=None, init=False, description="Server handle for running the FastAPI app.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to configure core FastAPI app and CORS settings.
        """
        
        # Initialize FastAPI app with title and description
        self.app = FastAPI(
            title=f"{self.service_name} API Server",
            description=self.description or self.service_name,
            lifespan=self.lifespan
        )

        # Configure CORS settings
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=self.cors_credentials,
            allow_methods=self.cors_methods,
            allow_headers=self.cors_headers,
        )
        
        logger.info(f"{self.service_name} FastAPI server initialized on port {self.service_port} with CORS settings.")

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
        Args:
            log_level (Optional[str]): The logging level for the Uvicorn server. Defaults to the global logging level.
        """
        # If log_level is not passed, fallback to the current logging level in the root logger
        if log_level is None:
            log_level = logging.getLevelName(logger.getEffectiveLevel()).lower()

        config = uvicorn.Config(self.app, host=self.service_host, port=self.service_port, log_level=log_level)
        self.server: uvicorn.Server = uvicorn.Server(config)

        # Set up signal handling for graceful shutdown
        loop = asyncio.get_event_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, lambda: asyncio.create_task(self.stop()))

        logger.info(f"Starting {self.service_name} service at {self.service_host}:{self.service_port}")
        await self.server.serve()

    async def stop(self):
        """
        Stop the FastAPI server gracefully.
        """
        if self.server:
            logger.info(f"Stopping {self.service_name} server on port {self.service_port}.")
            self.server.should_exit = True