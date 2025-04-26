from dapr_agents.service.fastapi.base import FastAPIServerBase
from dapr.ext.fastapi import DaprApp
from pydantic import Field
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class DaprFastAPIServer(FastAPIServerBase):
    """
    A Dapr-enabled service class extending FastAPIServerBase with Dapr-specific functionality.
    """

    # Initialized in model_post_init
    dapr_app: Optional[DaprApp] = Field(
        default=None, init=False, description="DaprApp for pub/sub integration."
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to configure the FastAPI app and Dapr-specific settings.
        """
        # Initialize inherited FastAPI app setup
        super().model_post_init(__context)

        # Initialize DaprApp for pub/sub
        self.dapr_app = DaprApp(self.app)

        logger.info(f"{self.service_name} DaprFastAPIServer initialized.")
