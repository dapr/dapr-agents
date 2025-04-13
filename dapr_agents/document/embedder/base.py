from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Any


class EmbedderBase(BaseModel, ABC):
    """
    Abstract base class for Embedders.
    """

    @abstractmethod
    def embed(self, query: str, **kwargs) -> List[Any]:
        """
        Search for content based on a query.

        Args:
            query (str): The search query.
            **kwargs: Additional search parameters.

        Returns:
            List[Any]: A list of results.
        """
        pass
