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
