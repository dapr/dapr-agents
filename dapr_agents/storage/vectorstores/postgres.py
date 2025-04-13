from dapr_agents.storage.vectorstores import VectorStoreBase
from dapr_agents.document.embedder import SentenceTransformerEmbedder
from dapr_agents.document.embedder.base import EmbedderBase
from typing import List, Dict, Optional, Iterable, Any, Literal, Union
from pydantic import Field, ConfigDict
import uuid
import logging

logger = logging.getLogger(__name__)


class PostgresVectorStore(VectorStoreBase):
    """
    A PostgreSQL-based vector store implementation leveraging pgvector for similarity search.
    Supports automatic embedding generation, metadata filtering, and flexible indexing.
    """

    connection_string: str = Field(..., description="PostgreSQL connection string.")
    table_name: str = Field(
        "vector_store", description="The table name for storing vectors."
    )
    embedding_dim: Optional[int] = Field(
        None, description="Fixed dimensionality of embedding vectors (optional)."
    )
    embedding_function: Optional[EmbedderBase] = Field(
        default_factory=SentenceTransformerEmbedder,
        description="Embedding function for embedding generation.",
    )
    pool_config: Dict = Field(
        default_factory=lambda: {"min_size": 1, "max_size": 10, "timeout": 30},
        description="Connection pool settings.",
    )
    index_type: Literal["hnsw", "ivfflat", "flat"] = Field(
        "ivfflat", description="The type of index to use for vector search."
    )
    index_params: Dict = Field(
        default_factory=lambda: {"lists": 100, "m": 16, "ef_construction": 64},
        description="Parameters for the vector index.",
    )

    pool: Optional[Any] = Field(
        default=None, init=False, description="Connection pool for PostgreSQL."
    )
    tracked_dimensions: set = Field(
        default_factory=set,
        init=False,
        description="Set of tracked vector dimensions for partial indexing.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the PostgreSQL connection pool, ensures table and initial index setup.
        """
        super().model_post_init(__context)

        try:
            from psycopg_pool import ConnectionPool
            from pgvector.psycopg import register_vector
        except ImportError as e:
            raise ImportError(
                "The psycopg, psycopg-pool, and pgvector libraries are required to use this store. "
                "Install them using pip install 'psycopg[binary,pool]' pgvector"
            ) from e

        try:
            self.pool: ConnectionPool = ConnectionPool(
                self.connection_string, **self.pool_config
            )
            with self.pool.connection() as conn:
                # Enable the pgvector extension if not already enabled
                with conn.cursor() as cursor:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    logger.info("pgvector extension ensured in the database.")

                # Register vector type with psycopg
                register_vector(conn)

                # Create the table
                with conn.cursor() as cursor:
                    embedding_column = (
                        f"VECTOR({self.embedding_dim})"
                        if self.embedding_dim
                        else "VECTOR"
                    )
                    cursor.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id UUID PRIMARY KEY,
                            document TEXT,
                            metadata JSONB,
                            embedding {embedding_column}
                        );
                    """
                    )
                    logger.info(f"Table '{self.table_name}' ensured.")

                # Create global index for fixed-dimension embeddings
                if self.embedding_dim:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            f"""
                            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                            ON {self.table_name} USING {self.index_type} (embedding vector_cosine_ops)
                            WITH ({", ".join(f"{k}={v}" for k, v in self.index_params.items())});
                            """
                        )
                        logger.info(
                            "Global index created for fixed-dimension embeddings."
                        )
                else:
                    logger.info(
                        "No fixed dimension specified; relying on dynamic partial indexing."
                    )
        except Exception as e:
            logger.error(f"Failed to initialize PostgresVectorStore: {e}")
            raise

    def _ensure_partial_index(self, dimension: int):
        """
        Creates a partial index for embeddings with the specified dimension if it doesn't already exist.
        """
        if dimension in self.tracked_dimensions:
            return

        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_{dimension}_idx
                        ON {self.table_name} USING hnsw ((embedding::vector({dimension})) vector_cosine_ops)
                        WHERE vector_dims(embedding) = {dimension};
                        """
                    )
                    logger.info(
                        f"Partial index created for embedding dimension {dimension}."
                    )
            self.tracked_dimensions.add(dimension)
        except Exception as e:
            logger.error(
                f"Failed to create partial index for dimension {dimension}: {e}"
            )

    def add(
        self,
        documents: Iterable[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        upsert: bool = False,
    ) -> List[str]:
        """
        Adds or upserts documents into the vector store.

        Args:
            documents (Iterable[str]): The documents to add or upsert.
            embeddings (Optional[List[List[float]]]): Precomputed embeddings.
            metadatas (Optional[List[Dict]]): Metadata for each document.
            ids (Optional[List[str]]): Unique IDs for the documents.
            upsert (bool): Whether to update existing records on conflict. Defaults to False.

        Returns:
            List[str]: List of IDs for the added or upserted documents.
        """

        try:
            from psycopg.types.json import Jsonb
        except ImportError as e:
            raise ImportError("Required library 'psycopg' is missing.") from e

        try:
            if embeddings is None:
                embeddings = self.embedding_function(list(documents))
                logger.info(
                    "Generated embeddings using the provided embedding function."
                )

            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]

            on_conflict_action = (
                """
                DO UPDATE SET
                    document = EXCLUDED.document,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """
                if upsert
                else "DO NOTHING"
            )

            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    for i, doc in enumerate(documents):
                        dimension = len(embeddings[i])
                        if not self.embedding_dim:
                            self._ensure_partial_index(dimension)

                        metadata = Jsonb(metadatas[i]) if metadatas else Jsonb({})
                        query = f"""
                        INSERT INTO {self.table_name} (id, document, metadata, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) {on_conflict_action};
                        """
                        cursor.execute(query, (ids[i], doc, metadata, embeddings[i]))
            logger.info(
                f"{'Upserted' if upsert else 'Added'} {len(documents)} documents."
            )
            return ids
        except Exception as e:
            logger.error(f"Failed to {'upsert' if upsert else 'add'} documents: {e}")
            raise

    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """
        Updates existing documents in the vector store.

        Args:
            ids (List[str]): A list of document IDs to update.
            embeddings (Optional[List[List[float]]]): The new embedding vectors for the documents.
            metadatas (Optional[List[Dict]]): The new metadata for the documents.
            documents (Optional[List[str]]): The new text for the documents.

        Returns:
            None
        """
        try:
            from psycopg.types.json import Jsonb
        except ImportError as e:
            raise ImportError("Required library 'psycopg' is missing.") from e

        try:
            if not any([embeddings, metadatas, documents]):
                raise ValueError(
                    "At least one of embeddings, metadatas, or documents must be provided for update."
                )

            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    for i, doc_id in enumerate(ids):
                        update_fields = []
                        params = []

                        if embeddings and embeddings[i]:
                            dimension = len(embeddings[i])
                            if not self.embedding_dim:
                                self._ensure_partial_index(dimension)
                            update_fields.append("embedding = %s")
                            params.append(embeddings[i])

                        if documents and documents[i]:
                            update_fields.append("document = %s")
                            params.append(documents[i])

                        if metadatas and metadatas[i]:
                            update_fields.append("metadata = %s")
                            params.append(Jsonb(metadatas[i]))

                        if not update_fields:
                            continue

                        params.append(doc_id)  # Add the ID for the WHERE clause

                        query = f"""
                        UPDATE {self.table_name}
                        SET {", ".join(update_fields)}
                        WHERE id = %s;
                        """
                        cursor.execute(query, params)

            logger.info(f"Updated {len(ids)} documents in {self.table_name}.")
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            raise

    def delete(self, ids: List[str]) -> bool:
        """
        Deletes documents from the vector store by their IDs.

        Args:
            ids (List[str]): List of document IDs to delete.

        Returns:
            bool: True if the deletion was successful, False otherwise.
        """
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"DELETE FROM {self.table_name} WHERE id = ANY(%s)", (ids,)
                    )
            logger.info(f"Deleted {len(ids)} documents.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def get(
        self, ids: Optional[List[str]] = None, with_embedding: bool = False
    ) -> List[Dict]:
        """
        Retrieves items from the PostgreSQL vector store by IDs. If no IDs are provided, retrieves all items.

        Args:
            ids (Optional[List[str]]): The IDs of the items to retrieve. If None, retrieves all items.
            with_embedding (bool): Whether to include embeddings in the retrieved data.

        Returns:
            List[Dict]: A list of dictionaries containing the metadata, documents, and optionally embeddings of the retrieved items.
        """
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    if ids:
                        query = "SELECT id, document, metadata"
                        if with_embedding:
                            query += ", embedding"
                        query += f" FROM {self.table_name} WHERE id = ANY(%s)"
                        cursor.execute(query, (ids,))
                    else:
                        query = "SELECT id, document, metadata"
                        if with_embedding:
                            query += ", embedding"
                        query += f" FROM {self.table_name}"
                        cursor.execute(query)

                    # Get column names for row-to-dict conversion
                    colnames = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()

            return [dict(zip(colnames, row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise

    def search_similar(
        self,
        query_texts: Optional[Union[str, List[str]]] = None,
        query_embeddings: Optional[Union[List[float], List[List[float]]]] = None,
        k: int = 4,
        distance_metric: str = "cosine",
        metadata_filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Perform a similarity search in the vector store with optional metadata filtering.

        Args:
            query_texts (Optional[Union[str, List[str]]]): Text queries to embed and search.
                If provided, embeddings are generated using the configured embedding function.
            query_embeddings (Optional[Union[List[float], List[List[float]]]]): Precomputed embeddings
                for similarity search. Provide either `query_texts` or `query_embeddings`, not both.
            k (int): Number of top results to return. Defaults to 4.
            distance_metric (str): Distance metric to use for similarity computation.
                Supported values are:
                    - "cosine": Cosine similarity.
                    - "l2": Euclidean distance.
                    - "inner_product": Inner product similarity.
                Defaults to "cosine".
            metadata_filter (Optional[Dict]): Metadata conditions for filtering results.
                Keys are metadata fields, and values are the required values for those fields.

        Returns:
            List[Dict]: A list of dictionaries, each representing a search result. Each dictionary contains:
                - "id" (str): The ID of the matched item.
                - "document" (str): The document content of the matched item.
                - "metadata" (Dict): Metadata associated with the matched item.
                - "similarity" (float): The similarity score of the matched item.

        Raises:
            ValueError: If neither `query_texts` nor `query_embeddings` is provided, or if both are provided.
            Exception: For any issues during the database query execution.
        """
        try:
            # Validate inputs
            if not query_texts and not query_embeddings:
                raise ValueError(
                    "Either `query_texts` or `query_embeddings` must be provided."
                )
            if query_texts and query_embeddings:
                raise ValueError(
                    "Provide either `query_texts` or `query_embeddings`, not both."
                )

            # Generate embeddings if query_texts is provided
            if query_texts:
                if isinstance(query_texts, str):
                    query_texts = [query_texts]
                query_embeddings = self.embedding_function(query_texts)

            # Handle single embedding case
            if isinstance(query_embeddings[0], (int, float)):
                query_embeddings = [query_embeddings]

            # Map distance metrics to PostgreSQL operators
            operator_map = {
                "cosine": "<=>",  # Cosine similarity
                "l2": "<->",  # Euclidean distance
                "inner_product": "<#>",  # Inner product
            }
            operator = operator_map.get(distance_metric, "<=>")

            results = []
            with self.pool.connection() as conn:
                # Use a compatible row factory or default cursor
                with conn.cursor() as cursor:
                    for embedding in query_embeddings:
                        # Convert the embedding to a PostgreSQL-compatible vector format
                        embedding_vector = f"ARRAY{embedding}::vector"

                        # Base query
                        query = f"""
                        SELECT id, document, metadata, 1 - (embedding {operator} {embedding_vector}) AS similarity
                        FROM {self.table_name}
                        """

                        # Add metadata filtering conditions if provided
                        params = []
                        if metadata_filter:
                            filter_conditions = " AND ".join(
                                ["metadata ->> %s = %s" for _ in metadata_filter]
                            )
                            query += f" WHERE {filter_conditions}"
                            params.extend(
                                sum(([k, v] for k, v in metadata_filter.items()), [])
                            )

                        # Add ordering and limit
                        query += " ORDER BY similarity DESC LIMIT %s"
                        params.append(k)

                        # Execute the query
                        cursor.execute(query, tuple(params))
                        for row in cursor.fetchall():
                            results.append(row)

            # Format results for consistency
            formatted_results = [
                {
                    "id": row[0],
                    "document": row[1],
                    "metadata": row[2],
                    "similarity": row[3],
                }
                for row in results
            ]

            return formatted_results
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise

    def reset(self):
        """
        Resets the vector store by truncating the table, deleting all stored data.

        Returns:
            None
        """
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"TRUNCATE TABLE {self.table_name}")
            logger.info(f"Table '{self.table_name}' reset.")
        except Exception as e:
            logger.error(f"Failed to reset table: {e}")
            raise

    def count(self) -> int:
        """
        Counts the number of documents in the vector store.

        Returns:
            int: The total number of documents in the store.
        """
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                    count = cursor.fetchone()[0]
            return count
        except Exception as e:
            logger.error(f"Failed to count documents in the store: {e}")
            raise
