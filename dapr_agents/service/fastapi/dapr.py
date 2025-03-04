from dapr_agents.service.fastapi.base import FastAPIServerBase
from dapr.conf import settings as dapr_settings
from dapr.ext.fastapi import DaprApp
from pydantic import Field
from typing import Optional, Any
import os
import logging

logger = logging.getLogger(__name__)

class DaprFastAPIServer(FastAPIServerBase):
    """
    A Dapr-enabled service class extending FastAPIServerBase with Dapr-specific functionality.
    """
    daprGrpcAddress: Optional[str] = Field(None, description="Full address of the Dapr sidecar.")
    daprGrpcHost: Optional[str] = Field(None, description="Host address for the Dapr gRPC endpoint.")
    daprGrpcPort: Optional[int] = Field(None, description="Port number for the Dapr gRPC endpoint.")

    # Initialized in model_post_init
    dapr_app: Optional[DaprApp] = Field(default=None, init=False, description="DaprApp for pub/sub integration.")

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to configure the FastAPI app and Dapr-specific settings.
        """
        # Initialize inherited FastAPI app setup
        super().model_post_init(__context)

        # Configure Dapr gRPC settings, using environment variables if provided
        env_daprGrpcHost = os.getenv('DAPR_RUNTIME_HOST')
        env_daprGrpcPort = os.getenv('DAPR_GRPC_PORT')

        # Resolve final values and persist them in the instance
        self.daprGrpcHost = self.daprGrpcHost or env_daprGrpcHost or dapr_settings.DAPR_RUNTIME_HOST
        self.daprGrpcPort = int(self.daprGrpcPort or env_daprGrpcPort or dapr_settings.DAPR_GRPC_PORT)
        
        # Set the Dapr gRPC address based on finalized settings
        self.daprGrpcAddress = self.daprGrpcAddress or f"{self.daprGrpcHost}:{self.daprGrpcPort}"

        # Initialize DaprApp for pub/sub and DaprClient for state and messaging
        self.dapr_app = DaprApp(self.app)

        logger.info(f"{self.service_name} DaprFastAPIServer initialized.")
        logger.info(f"Dapr gRPC address: {self.daprGrpcAddress}.")