from dapr_agents.storage.vectorstores import VectorStoreBase
from dapr_agents.document.embedder.base import EmbedderBase
from typing import List, Dict, Optional, Iterable, Any, Union
from pydantic import Field, ConfigDict
import uuid
import logging

logger = logging.getLogger(__name__)


class RedisVectorStore(VectorStoreBase):
    """
    Redis-based vector store implementation using RedisVL for similarity search.
    Supports storing, querying, and filtering documents with embeddings.
    """

    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL.",
    )
    index_name: str = Field(
        default="dapr_agents",
        description="The name of the Redis search index.",
    )
    embedding_function: EmbedderBase = Field(
        ...,
        description="Embedding function for embedding generation.",
    )
    embedding_dims: int = Field(
        default=384,
        description="Dimensionality of the embedding vectors.",
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric for similarity search (cosine, l2, ip).",
    )
    storage_type: str = Field(
        default="hash",
        description="Redis storage type (hash or json).",
    )

    index: Optional[Any] = Field(
        default=None, init=False, description="RedisVL SearchIndex instance."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization setup for RedisVectorStore.
        Creates the search index with the specified schema.
        """
        try:
            from redisvl.index import SearchIndex
            from redisvl.schema import IndexSchema
        except ImportError:
            raise ImportError(
                "The `redisvl` library is required to use this store. "
                "Install it using `pip install redisvl`."
            )

        schema_dict = {
            "index": {
                "name": self.index_name,
                "prefix": f"{self.index_name}_doc",
                "storage_type": self.storage_type,
            },
            "fields": [
                {"name": "doc_id", "type": "tag"},
                {"name": "document", "type": "text"},
                {"name": "metadata", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": self.embedding_dims,
                        "algorithm": "flat",
                        "distance_metric": self.distance_metric,
                    },
                },
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        self.index = SearchIndex(schema, redis_url=self.redis_url)

        # Create index if it doesn't exist
        if not self.index.exists():
            self.index.create(overwrite=False)
            logger.info(f"RedisVectorStore index '{self.index_name}' created.")
        else:
            logger.info(f"RedisVectorStore connected to existing index '{self.index_name}'.")

        super().model_post_init(__context)

    def add(
        self,
        documents: Iterable[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Adds documents and their corresponding metadata to the Redis index.

        Args:
            documents (Iterable[str]): The documents to add.
            embeddings (Optional[List[List[float]]]): The embeddings of the documents.
                If None, the configured embedding function will generate embeddings.
            metadatas (Optional[List[dict]]): The metadata associated with each document.
            ids (Optional[List[str]]): The IDs for each document.
                If not provided, random UUIDs are generated.

        Returns:
            List[str]: List of IDs for the added documents.
        """
        try:
            from redisvl.redis.utils import array_to_buffer
        except ImportError:
            raise ImportError(
                "The `redisvl` library is required. Install it using `pip install redisvl`."
            )

        try:
            documents_list = list(documents)

            if embeddings is None:
                embeddings = self.embedding_function(documents_list)
                logger.debug("Generated embeddings using the embedding function.")

            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents_list]

            if metadatas is None:
                metadatas = [{} for _ in documents_list]

            data = []
            for i, doc in enumerate(documents_list):
                record = {
                    "doc_id": ids[i],
                    "document": doc,
                    "metadata": str(metadatas[i]),
                    "embedding": array_to_buffer(embeddings[i], dtype="float32"),
                }
                data.append(record)

            self.index.load(data, id_field="doc_id")
            logger.info(f"Added {len(documents_list)} documents to RedisVectorStore.")
            return ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def delete(self, ids: List[str]) -> Optional[bool]:
        """
        Deletes documents from the Redis index by their IDs.

        Args:
            ids (List[str]): The IDs of the documents to delete.

        Returns:
            Optional[bool]: True if deletion was successful, False otherwise.
        """
        try:
            from redis import Redis
        except ImportError:
            raise ImportError(
                "The `redis` library is required. Install it using `pip install redis`."
            )

        try:
            client = Redis.from_url(self.redis_url)
            prefix = f"{self.index_name}_doc"

            deleted_count = 0
            for doc_id in ids:
                key = f"{prefix}:{doc_id}"
                result = client.delete(key)
                deleted_count += result

            logger.info(f"Deleted {deleted_count} documents from RedisVectorStore.")
            return deleted_count > 0

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def get(self, ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieves documents from the Redis index by IDs.
        If no IDs are provided, retrieves all documents.

        Args:
            ids (Optional[List[str]]): The IDs of the documents to retrieve.
                If None, retrieves all documents.

        Returns:
            List[Dict]: A list of dictionaries containing document data.
        """
        try:
            import ast

            results = []

            if ids is not None:
                for doc_id in ids:
                    doc = self.index.fetch(doc_id)
                    if doc:
                        metadata = doc.get("metadata", "{}")
                        try:
                            metadata = ast.literal_eval(metadata)
                        except (ValueError, SyntaxError):
                            metadata = {}
                        results.append({
                            "id": doc.get("doc_id", doc_id),
                            "document": doc.get("document", ""),
                            "metadata": metadata,
                        })
            else:
                # Get all documents by querying with a match-all
                from redisvl.query import FilterQuery

                query = FilterQuery(
                    return_fields=["doc_id", "document", "metadata"],
                    num_results=10000,  # Large number to get all
                )
                docs = self.index.query(query)
                for doc in docs:
                    metadata = doc.get("metadata", "{}")
                    try:
                        metadata = ast.literal_eval(metadata)
                    except (ValueError, SyntaxError):
                        metadata = {}
                    results.append({
                        "id": doc.get("doc_id", ""),
                        "document": doc.get("document", ""),
                        "metadata": metadata,
                    })

            return results

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise

    def reset(self):
        """
        Resets the Redis vector store by clearing all data.
        The index structure is preserved.
        """
        try:
            self.index.clear()
            logger.info(f"RedisVectorStore index '{self.index_name}' cleared.")
        except Exception as e:
            logger.error(f"Failed to reset RedisVectorStore: {e}")
            raise

    def search_similar(
        self,
        query_texts: Optional[Union[List[str], str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Performs a similarity search in the Redis index.

        Args:
            query_texts (Optional[Union[List[str], str]]): The query texts.
            query_embeddings (Optional[List[List[float]]]): The query embeddings.
            k (int): The number of results to return.

        Returns:
            List[Dict]: A list of dictionaries containing the search results.
        """
        try:
            from redisvl.query import VectorQuery
        except ImportError:
            raise ImportError(
                "The `redisvl` library is required. Install it using `pip install redisvl`."
            )

        try:
            import ast

            if query_texts is not None:
                if isinstance(query_texts, str):
                    query_texts = [query_texts]
                query_embeddings = self.embedding_function(query_texts)

            if query_embeddings is None:
                raise ValueError(
                    "Either query_texts or query_embeddings must be provided."
                )

            # Handle single embedding
            if isinstance(query_embeddings[0], (int, float)):
                query_embeddings = [query_embeddings]

            all_results = []
            for embedding in query_embeddings:
                query = VectorQuery(
                    vector=embedding,
                    vector_field_name="embedding",
                    return_fields=["doc_id", "document", "metadata"],
                    num_results=k,
                )
                results = self.index.query(query)

                for doc in results:
                    metadata = doc.get("metadata", "{}")
                    try:
                        metadata = ast.literal_eval(metadata)
                    except (ValueError, SyntaxError):
                        metadata = {}
                    all_results.append({
                        "id": doc.get("doc_id", ""),
                        "document": doc.get("document", ""),
                        "metadata": metadata,
                        "vector_distance": doc.get("vector_distance", 0.0),
                    })

            return all_results

        except Exception as e:
            logger.error(f"An error occurred during similarity search: {e}")
            return []

    def count(self) -> int:
        """
        Counts the number of documents in the Redis index.

        Returns:
            int: The number of documents in the index.
        """
        try:
            info = self.index.info()
            return int(info.get("num_docs", 0))
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
