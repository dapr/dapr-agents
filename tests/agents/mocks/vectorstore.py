from dapr_agents.storage import VectorStoreBase
from typing import List, Dict, Any, Optional
from unittest.mock import Mock
from pydantic import PrivateAttr


class MockVectorStore(VectorStoreBase):
    """Mock vector store for testing."""

    _embed_documents = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]])

    def embed_documents(self, *args, **kwargs):
        return self._embed_documents(*args, **kwargs)

    def add(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Mock implementation of add."""
        return ids or [f"mock_id_{i}" for i in range(len(documents))]

    def delete(self, ids: List[str]) -> None:
        """Mock implementation of delete."""
        pass

    def get(
        self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Mock implementation of get."""
        return {"documents": [], "metadatas": [], "embeddings": [], "ids": []}

    def reset(self) -> None:
        """Mock implementation of reset."""
        pass

    def search_similar(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mock implementation of search_similar."""
        return {
            "documents": [],
            "metadatas": [],
            "embeddings": [],
            "ids": [],
            "distances": [],
        }
