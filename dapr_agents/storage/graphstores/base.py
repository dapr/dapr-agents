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
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)


class GraphStoreBase(BaseModel, ABC):
    """
    Base interface for a graph store.
    """

    client: Any = Field(..., description="The client to interact with the graph store.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def add_node(self, label: str, properties: Dict[str, Any]) -> None:
        """Add a node to the graph store.

        Args:
            label (str): The label of the node.
            properties (Dict[str, Any]): The properties of the node.
        """
        pass

    @abstractmethod
    def add_relationship(
        self,
        start_node_props: Dict[str, Any],
        end_node_props: Dict[str, Any],
        relationship_type: str,
        relationship_props: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a relationship to the graph store.

        Args:
            start_node_props (Dict[str, Any]): The properties of the start node.
            end_node_props (Dict[str, Any]): The properties of the end node.
            relationship_type (str): The type of the relationship.
            relationship_props (Optional[Dict[str, Any]]): The properties of the relationship.
        """
        pass

    @abstractmethod
    def query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a query against the graph store.

        Args:
            query (str): The query to execute.
            params (Dict[str, Any], optional): The parameters for the query.

        Returns:
            List[Dict[str, Any]]: The query results.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the graph store."""
        pass
