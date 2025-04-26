from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Optional


class APIServerBase(BaseModel, ABC):
    """
    Abstract base class for API server services.
    Supports both FastAPI and Flask implementations.
    """

    service_name: str = Field(..., description="The name of the API service.")
    service_port: Optional[int] = Field(
        default=None,
        description="Port to run the API server on. If None, use a random available port.",
    )
    service_host: str = Field("0.0.0.0", description="Host address for the API server.")

    @abstractmethod
    async def start(self, log_level=None):
        """
        Abstract method to start the API server.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def stop(self):
        """
        Abstract method to stop the API server.
        Must be implemented by subclasses.
        """
        pass
