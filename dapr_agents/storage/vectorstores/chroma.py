from dapr_agents.storage.vectorstores import VectorStoreBase
from dapr_agents.document.embedder.base import EmbedderBase
from typing import List, Dict, Optional, Iterable, Union, Any
from pydantic import Field, ConfigDict
import uuid
import logging

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStoreBase):
    """
    Chroma-based vector store implementation with flexible persistence and server mode.
    Supports storing, querying, and filtering documents with embeddings generated on-the-fly.
    """

    name: str = Field(
        default="dapr_agents", description="The name of the Chroma collection."
    )
    api_key: Optional[str] = Field(
        None, description="API key for the embedding service."
    )
    embedding_function: EmbedderBase = Field(
        ...,  # Required field, no default
        description="Embedding function for embedding generation.",
    )
    persistent: bool = Field(False, description="Whether to enable persistent storage.")
    path: Optional[str] = Field(None, description="Path for persistent storage.")
    client_server_mode: bool = Field(
        False, description="Whether to enable client-server mode."
    )
    host: str = Field(
        "localhost", description="Host for the Chroma server in client-server mode."
    )
    port: int = Field(
        8000, description="Port for the Chroma server in client-server mode."
    )
    settings: Optional[Any] = Field(
        None, description="Optional Chroma settings object."
    )

    client: Optional[Any] = Field(
        default=None, init=False, description="Chroma client instance."
    )
    collection: Optional[Any] = Field(
        default=None, init=False, description="Chroma collection for document storage."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization setup for ChromaVectorStore, configuring the embedding manager, client, and collection.
        """
        try:
            from chromadb import Client
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "The `chromadb` library is required to use this store. "
                "Install it using `pip install chromadb`."
            )

        if not self.settings:
            # Start with base settings
            settings_kwargs = {"allow_reset": True, "anonymized_telemetry": False}

            # Add specific settings based on the configuration
            if self.client_server_mode:
                settings_kwargs.update(
                    {
                        "chroma_server_host": self.host,
                        "chroma_server_http_port": self.port,
                        "chroma_api_impl": "chromadb.api.fastapi.FastAPI",
                    }
                )
            elif self.persistent:
                settings_kwargs.update(
                    {
                        "persist_directory": self.path or "db",
                        "is_persistent": True,
                    }
                )

            # Initialize settings
            self.settings = ChromaSettings(**settings_kwargs)

        # Initialize Chroma client and collection
        self.client: Client = Client(settings=self.settings)
        self.collection = self.client.get_or_create_collection(
            name=self.name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"ChromaVectorStore initialized with collection: {self.name}")
        super().model_post_init(__context)

    def add(
        self,
        documents: Iterable[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Adds documents and their corresponding metadata to the Chroma collection.

        Args:
            documents (Iterable[str]): The documents to add to the collection.
            embeddings (Optional[List[List[float]]]): The embeddings of the documents to add to the collection.
                If None, the configured embedding function will automatically generate embeddings.
            metadatas (Optional[List[dict]]): The metadata associated with each text.
            ids (Optional[List[str]]): The IDs for each text. If not provided, random UUIDs are generated.
        """
        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]

            self.collection.add(
                documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
            )
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def delete(self, ids: Optional[List[int]] = None) -> Optional[bool]:
        """
        Deletes items from the Chroma collection by IDs.

        Args:
            ids (Optional[List[int]]): The IDs of the items to delete.

        Returns:
            Optional[bool]: True if deletion is successful, False otherwise.
        """
        if ids:
            string_ids = [str(i) for i in ids]
            self.collection.delete(ids=string_ids)
            return True
        return False

    def get(
        self,
        ids: Optional[List[str]] = None,
        include: Optional[List[str]] = ["documents", "metadatas"],
    ) -> List[Dict]:
        """
        Retrieves items from the Chroma collection by IDs. If no IDs are provided, retrieves all items.

        Args:
            ids (Optional[List[str]]): The IDs of the items to retrieve. If None, retrieves all items.
            include (Optional[List[str]]): List of fields to include in the response.

        Returns:
            List[Dict]: A list of dictionaries containing the metadata and documents of the retrieved items.
        """
        if ids is None:
            items = self.collection.get(include=include)
        else:
            items = self.collection.get(ids=ids, include=include)

        return [
            {"id": item_id, "metadata": item_meta, "document": item_doc}
            for item_id, item_meta, item_doc in zip(
                items["ids"], items["metadatas"], items["documents"]
            )
        ]

    def reset(self):
        """Resets the Chroma database."""
        self.client.reset()

    def update(
        self,
        ids: List[str],
        metadatas: Optional[List[dict]] = None,
        documents: Optional[List[str]] = None,
    ):
        """
        Updates items in the Chroma collection.

        Args:
            ids (List[str]): The IDs of the items to update.
            metadatas (Optional[List[dict]]): The new metadata for the items.
            documents (Optional[List[str]]): The new documents for the items.
        """
        self.collection.update(ids=ids, metadatas=metadatas, documents=documents)

    def count(self) -> int:
        """
        Counts the number of items in the Chroma collection.

        Returns:
            int: The number of items in the collection.
        """
        return self.collection.count()

    def search_similar(
        self,
        query_texts: Optional[Union[List[str], str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        k: int = 4,
    ) -> List[Dict]:
        """
        Performs a similarity search in the Chroma collection using either query texts or query embeddings.

        Args:
            query_texts (Optional[Union[List[str], str]]): The query texts.
            query_embeddings (Optional[List[List[float]]]): The query embeddings.
            k (int): The number of results to return.

        Returns:
            List[Dict]: A list of dictionaries containing the metadata of the most similar documents.
        """
        try:
            if query_texts:
                results = self.collection.query(query_texts=query_texts, n_results=k)
            elif query_embeddings:
                results = self.collection.query(
                    query_embeddings=query_embeddings, n_results=k
                )
            else:
                raise ValueError(
                    "Either query_texts or query_embeddings must be provided."
                )
            return results
        except Exception as e:
            logger.error(f"An error occurred during similarity search: {e}")
            return []

    def query_with_filters(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        k: int = 4,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Queries the Chroma collection with additional filters using either query texts or query embeddings.

        Args:
            query_texts (Optional[List[str]]): The query texts.
            query_embeddings (Optional[List[List[float]]]): The query embeddings.
            k (int): The number of results to return.
            where (Optional[Dict]): Criteria to filter items.

        Returns:
            List[Dict]: A list of dictionaries containing the metadata of the most similar documents.
        """
        try:
            if query_texts is not None:
                results = self.collection.query(
                    query_texts=query_texts,
                    n_results=k,
                    where=where,
                    include=["distances", "documents", "metadatas"],
                )
            elif query_embeddings is not None:
                results = self.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=k,
                    where=where,
                    include=["distances", "documents", "metadatas"],
                )
            else:
                raise ValueError(
                    "Either query_texts or query_embeddings must be provided."
                )
            return results
        except Exception as e:
            logger.error(f"An error occurred during filtered query search: {e}")
            return []
