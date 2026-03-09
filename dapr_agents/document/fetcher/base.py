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
from pydantic import BaseModel
from typing import List, Any


class FetcherBase(BaseModel, ABC):
    """
    Abstract base class for fetchers.
    """

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Any]:
        """
        Search for content based on a query.

        Args:
            query (str): The search query.
            **kwargs: Additional search parameters.

        Returns:
            List[Any]: A list of results.
        """
        pass
