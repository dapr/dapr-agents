#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
