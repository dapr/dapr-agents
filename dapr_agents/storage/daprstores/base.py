from dapr.clients import DaprClient
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any


class DaprStoreBase(BaseModel):
    """
    Pydantic-based Dapr store base model with configuration options for store name, address, host, and port.
    """

    store_name: str = Field(..., description="The name of the Dapr store.")
    client: Optional[DaprClient] = Field(
        default=None, init=False, description="Dapr client for store operations."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to set Dapr settings based on provided or environment values for host and port.
        """

        # Complete post-initialization
        super().model_post_init(__context)
