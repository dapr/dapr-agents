from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

class APIServerBase(BaseModel, ABC):
    """
    Abstract base class for API server services.
    Supports both FastAPI and Flask implementations.
    """
    
    service_name: str = Field(..., description="The name of the API service.")
    service_port: int = Field(..., description="The port number to run the API server on.")
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