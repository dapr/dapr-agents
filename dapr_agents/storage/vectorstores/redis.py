from dapr_agents.storage.vectorstores import VectorStoreBase
from dapr_agents.document.embedder.base import EmbedderBase
from typing import List, Dict, Literal, Optional, Iterable, Any, Union
from pydantic import Field, ConfigDict
import ast
import uuid
import logging

logger = logging.getLogger(__name__)

try:
    from redisvl.index import SearchIndex
    from redisvl.schema import IndexSchema
    from redisvl.query import VectorQuery, FilterQuery
    from redisvl.redis.utils import array_to_buffer
except ImportError:
    SearchIndex = None  # type: ignore
    IndexSchema = None  # type: ignore
    VectorQuery = None  # type: ignore
    FilterQuery = None  # type: ignore
    array_to_buffer = None  # type: ignore

try:
    from redis import Redis
except ImportError:
    Redis = None  # type: ignore

DOC_KEY_SUFFIX = "_doc"
EMBEDDING_DTYPE = "float32"
FIELD_DOC_ID = "doc_id"
FIELD_DOCUMENT = "document"
FIELD_METADATA = "metadata"
FIELD_EMBEDDING = "embedding"


class RedisVectorStore(VectorStoreBase):
    """
    Redis-based vector store implementation using RedisVL for similarity search.
    Supports storing, querying, and filtering documents with embeddings.
    """

    url: str = Field(
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
    embedding_dimensions: int = Field(
        default=384,
        description="Dimensionality of the embedding vectors.",
    )
    distance_metric: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description="Distance metric for similarity search. "
        "Based on Redis vector search supported distance metrics: "
        "https://redis.io/docs/latest/develop/ai/search-and-query/vectors/",
    )
    storage_type: Literal["hash", "json"] = Field(
        default="hash",
        description="Redis storage type (hash or json).",
    )

    search_index: Optional[Any] = Field(
        default=None, init=False, description="RedisVL SearchIndex instance."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _key_prefix(self) -> str:
        """Returns the key prefix used for storing documents in Redis."""
        return f"{self.index_name}{DOC_KEY_SUFFIX}"

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization setup for RedisVectorStore.
        Creates the search index with the specified schema.
        """
        if SearchIndex is None or IndexSchema is None:
            raise ImportError(
                "The `redisvl` library is required to use this store. "
                "Install it using `pip install redisvl`."
            )

        schema_dict = {
            "index": {
                "name": self.index_name,
                "prefix": self._key_prefix,
                "storage_type": self.storage_type,
            },
            "fields": [
                {"name": FIELD_DOC_ID, "type": "tag"},
                {"name": FIELD_DOCUMENT, "type": "text"},
                {"name": FIELD_METADATA, "type": "text"},
                {
                    "name": FIELD_EMBEDDING,
                    "type": "vector",
                    "attrs": {
                        "dims": self.embedding_dimensions,
                        "algorithm": "flat",
                        "distance_metric": self.distance_metric,
                    },
                },
            ],
        }

        schema = IndexSchema.from_dict(schema_dict)
        self.search_index = SearchIndex(schema, redis_url=self.url)

        if not self.search_index.exists():
            self.search_index.create(overwrite=False)
            logger.info(f"RedisVectorStore index '{self.index_name}' created.")
        else:
            logger.info(
                f"RedisVectorStore connected to existing index '{self.index_name}'."
            )

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
        if array_to_buffer is None:
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
                    FIELD_DOC_ID: ids[i],
                    FIELD_DOCUMENT: doc,
                    FIELD_METADATA: str(metadatas[i]),
                    FIELD_EMBEDDING: array_to_buffer(
                        embeddings[i], dtype=EMBEDDING_DTYPE
                    ),
                }
                data.append(record)

            self.search_index.load(data, id_field=FIELD_DOC_ID)
            logger.info(f"Added {len(documents_list)} documents to RedisVectorStore.")
            return ids

        except Exception as e:
            logger.exception(f"Failed to add documents: {e}")
            raise

    def delete(self, ids: List[str]) -> Optional[bool]:
        """
        Deletes documents from the Redis index by their IDs.

        Args:
            ids (List[str]): The IDs of the documents to delete.

        Returns:
            Optional[bool]: True if any documents were deleted, None on failure.
        """
        if Redis is None:
            raise ImportError(
                "The `redis` library is required. Install it using `pip install redis`."
            )

        try:
            client = Redis.from_url(self.url)

            deleted_count = 0
            for doc_id in ids:
                key = f"{self._key_prefix}:{doc_id}"
                result = client.delete(key)
                deleted_count += result

            logger.debug(f"Deleted {deleted_count} documents from RedisVectorStore.")
            return deleted_count > 0

        except Exception as e:
            logger.exception(f"Failed to delete documents: {e}")
            return None

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
        if FilterQuery is None:
            raise ImportError(
                "The `redisvl` library is required. Install it using `pip install redisvl`."
            )

        results = []

        try:
            if ids is not None:
                for doc_id in ids:
                    doc = self.search_index.fetch(doc_id)
                    if doc:
                        metadata = doc.get(FIELD_METADATA, "{}")
                        try:
                            metadata = ast.literal_eval(metadata)
                        except (ValueError, SyntaxError):
                            metadata = {}
                        results.append(
                            {
                                "id": doc.get(FIELD_DOC_ID, doc_id),
                                "document": doc.get(FIELD_DOCUMENT, ""),
                                "metadata": metadata,
                            }
                        )
            else:
                query = FilterQuery(
                    return_fields=[FIELD_DOC_ID, FIELD_DOCUMENT, FIELD_METADATA],
                    num_results=10000, 
                )
                docs = self.search_index.query(query)
                for doc in docs:
                    metadata = doc.get(FIELD_METADATA, "{}")
                    try:
                        metadata = ast.literal_eval(metadata)
                    except (ValueError, SyntaxError):
                        metadata = {}
                    results.append(
                        {
                            "id": doc.get(FIELD_DOC_ID, ""),
                            "document": doc.get(FIELD_DOCUMENT, ""),
                            "metadata": metadata,
                        }
                    )

            return results

        except Exception as e:
            logger.exception(f"Failed to retrieve documents: {e}")
            raise

    def reset(self):
        """
        Resets the Redis vector store by clearing all data.
        The index structure is preserved.
        """
        try:
            self.search_index.clear()
            logger.debug(f"RedisVectorStore index '{self.index_name}' cleared.")
        except Exception as e:
            logger.exception(f"Failed to reset RedisVectorStore: {e}")
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
        if VectorQuery is None:
            raise ImportError(
                "The `redisvl` library is required. Install it using `pip install redisvl`."
            )

        if query_texts is None and query_embeddings is None:
            raise ValueError(
                "Either query_texts or query_embeddings must be provided."
            )

        if query_texts is not None:
            if isinstance(query_texts, str):
                query_texts = [query_texts]
            query_embeddings = self.embedding_function(query_texts)

        # Handle single embedding
        if isinstance(query_embeddings[0], (int, float)):
            query_embeddings = [query_embeddings]

        try:
            all_results = []
            for embedding in query_embeddings:
                query = VectorQuery(
                    vector=embedding,
                    vector_field_name=FIELD_EMBEDDING,
                    return_fields=[FIELD_DOC_ID, FIELD_DOCUMENT, FIELD_METADATA],
                    num_results=k,
                )
                results = self.search_index.query(query)

                for doc in results:
                    metadata = doc.get(FIELD_METADATA, "{}")
                    try:
                        metadata = ast.literal_eval(metadata)
                    except (ValueError, SyntaxError):
                        metadata = {}
                    all_results.append(
                        {
                            "id": doc.get(FIELD_DOC_ID, ""),
                            "document": doc.get(FIELD_DOCUMENT, ""),
                            "metadata": metadata,
                            "vector_distance": doc.get("vector_distance", 0.0),
                        }
                    )

            return all_results

        except Exception as e:
            logger.exception(f"An error occurred during similarity search: {e}")
            return []

    def count(self) -> int:
        """
        Counts the number of documents in the Redis index.

        Returns:
            int: The number of documents in the index.
        """
        try:
            info = self.search_index.info()
            return int(info.get("num_docs", 0))
        except Exception as e:
            logger.exception(f"Failed to count documents: {e}")
            return 0
