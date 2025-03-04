from dapr.conf import settings as dapr_settings
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any
import os

class DaprStoreBase(BaseModel):
    """
    Pydantic-based Dapr store base model with configuration options for store name, address, host, and port.
    """

    store_name: str = Field(..., description="The name of the Dapr store.")
    daprGrpcAddress: Optional[str] = Field(None, description="The full address of the Dapr sidecar (host:port). If not provided, constructed from host and port.")
    daprGrpcHost: Optional[str] = Field(None, description="The host of the Dapr sidecar, defaults to environment variable or '127.0.0.1'.")
    daprGrpcPort: Optional[int] = Field(None, description="The port of the Dapr sidecar, defaults to environment variable or '50001'.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to set Dapr settings based on provided or environment values for host and port.
        """
        # Configure Dapr gRPC settings, using environment variables if provided
        env_daprGrpcHost = os.getenv('DAPR_RUNTIME_HOST')
        env_daprGrpcPort = os.getenv('DAPR_GRPC_PORT')

        # Resolve final values and persist them in the instance
        self.daprGrpcHost = self.daprGrpcHost or env_daprGrpcHost or dapr_settings.DAPR_RUNTIME_HOST
        self.daprGrpcPort = int(self.daprGrpcPort or env_daprGrpcPort or dapr_settings.DAPR_GRPC_PORT)
        
        # Set the Dapr gRPC address based on finalized settings
        self.daprGrpcAddress = self.daprGrpcAddress or f"{self.daprGrpcHost}:{self.daprGrpcPort}"

        # Complete post-initialization
        super().model_post_init(__context)