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

import pytest
import uuid

try:
    import redisvl  # noqa: F401
    from redis import Redis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from dapr_agents.document.embedder.sentence import SentenceTransformerEmbedder
    from dapr_agents.storage.vectorstores.redis import RedisVectorStore

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None  # type: ignore
    RedisConnectionError = ConnectionError  # Fallback to built-in ConnectionError
    SentenceTransformerEmbedder = None  # type: ignore
    RedisVectorStore = None  # type: ignore

pytestmark = pytest.mark.skipif(
    not REDIS_AVAILABLE,
    reason="redisvl or sentence-transformers not installed - optional dependencies",
)


@pytest.fixture
def redis_available():
    """Check if Redis is available before attempting to create store.

    This fixture checks Redis availability and skips tests if Redis is not available.
    """
    try:
        if Redis is not None:
            client = Redis.from_url("redis://localhost:6379")
            client.ping()
            client.close()
    except (RedisConnectionError, Exception) as e:
        pytest.skip(f"Redis not available: {e}")


class TestRedisVectorStore:
    """Test cases for RedisVectorStore."""

    @pytest.fixture
    def embedder(self, test_model_name):
        """Create a SentenceTransformerEmbedder fixture."""
        return SentenceTransformerEmbedder(model=test_model_name)

    @pytest.fixture
    def redis_index_name(self):
        """Create a unique index name for testing."""
        return f"test_redis_index_{uuid.uuid4().hex[:8]}"

    @pytest.fixture
    def vector_store(self, embedder, redis_index_name, redis_available):
        """Create a RedisVectorStore fixture.

        Note: This requires a running Redis instance with the RediSearch module.
        Tests will be skipped if Redis is not available.
        """

        try:
            store = RedisVectorStore(
                index_name=redis_index_name,
                embedding_function=embedder,
                embedding_dimensions=384,  # all-MiniLM-L6-v2 produces 384-dim embeddings
                url="redis://localhost:6379",
            )
            yield store
            # Cleanup after test
            try:
                store.search_index.delete(drop=True)
            except Exception as cleanup_error:
                pytest.fail(f"Failed to cleanup Redis index: {cleanup_error}")
        except Exception as e:
            # Only skip on connection errors, let other exceptions fail the test
            if isinstance(e, (RedisConnectionError, ConnectionError)):
                pytest.skip(f"Redis not available: {e}")
            raise

    def test_redis_vectorstore_creation(
        self, embedder, redis_index_name, redis_available
    ):
        """Test that RedisVectorStore can be created successfully."""

        try:
            vector_store = RedisVectorStore(
                index_name=redis_index_name,
                embedding_function=embedder,
                embedding_dimensions=384,
            )
            assert vector_store is not None
            assert vector_store.index_name == redis_index_name
            # Cleanup
            try:
                vector_store.search_index.delete(drop=True)
            except Exception as cleanup_error:
                pytest.fail(f"Failed to cleanup Redis index: {cleanup_error}")
        except Exception as e:
            # Only skip on connection errors, let other exceptions fail the test
            if isinstance(e, (RedisConnectionError, ConnectionError)):
                pytest.skip(f"Redis not available: {e}")
            raise

    def test_embedder_has_name_attribute(self, embedder):
        """Test that the embedder has a name attribute."""
        assert hasattr(embedder, "name"), "Embedder should have a name attribute"
        assert embedder.name is not None, "Name attribute should not be None"

    def test_vectorstore_with_embedder(self, vector_store, redis_index_name):
        """Test that RedisVectorStore works with the embedder."""
        assert vector_store is not None
        assert hasattr(vector_store, "index_name")
        assert vector_store.index_name == redis_index_name

    def test_vectorstore_different_names(self, embedder, redis_available):
        """Test creating vector stores with different names."""

        names = [f"test_index_{uuid.uuid4().hex[:8]}" for _ in range(3)]
        stores = []

        for name in names:
            try:
                vector_store = RedisVectorStore(
                    index_name=name,
                    embedding_function=embedder,
                    embedding_dimensions=384,
                )
                stores.append(vector_store)
                assert vector_store is not None
                assert vector_store.index_name == name
            except Exception as e:
                # Only skip on connection errors, let other exceptions fail the test
                if isinstance(e, (RedisConnectionError, ConnectionError)):
                    pytest.skip(f"Redis not available: {e}")
                raise

        # Cleanup
        for store in stores:
            try:
                store.search_index.delete(drop=True)
            except Exception as cleanup_error:
                pytest.fail(f"Failed to cleanup Redis index: {cleanup_error}")

    def test_vectorstore_distance_metrics(self, embedder, redis_available):
        """Test creating vector stores with different distance metrics."""

        metrics = ["cosine", "l2", "ip"]
        stores = []

        for i, metric in enumerate(metrics):
            try:
                vector_store = RedisVectorStore(
                    index_name=f"test_metric_{metric}_{uuid.uuid4().hex[:8]}",
                    embedding_function=embedder,
                    embedding_dimensions=384,
                    distance_metric=metric,
                )
                stores.append(vector_store)
                assert vector_store is not None
                assert vector_store.distance_metric == metric
            except Exception as e:
                # Only skip on connection errors, let other exceptions fail the test
                if isinstance(e, (RedisConnectionError, ConnectionError)):
                    pytest.skip(f"Redis not available: {e}")
                raise

        # Cleanup
        for store in stores:
            try:
                store.search_index.delete(drop=True)
            except Exception as cleanup_error:
                pytest.fail(f"Failed to cleanup Redis index: {cleanup_error}")

    def test_vectorstore_end_to_end(self, vector_store):
        """End-to-end test for add, get, search_similar, and delete."""
        # Prepare documents to index
        documents = [
            "Redis is a fast in-memory database.",
            "Vector search enables semantic similarity queries.",
        ]
        metadatas = [
            {"topic": "redis", "type": "database"},
            {"topic": "vector-search", "type": "ml"},
        ]

        # Add documents to the vector store
        added_ids = vector_store.add(documents, metadatas=metadatas)
        assert len(added_ids) == 2
        doc1_id = added_ids[0]

        # Run a similarity search that should match the first document
        results = vector_store.search_similar(
            query_texts="fast in-memory database",
            k=1,
        )
        assert results, "Expected at least one result from search_similar"
        assert len(results) >= 1

        # Retrieve the first document directly by ID
        fetched = vector_store.get([doc1_id])
        assert fetched, "Expected to fetch at least one document by ID"
        assert len(fetched) == 1
        assert fetched[0]["id"] == doc1_id
        assert fetched[0]["document"] == documents[0]
        assert fetched[0]["metadata"] == metadatas[0]

        # Delete the first document and verify it is no longer retrievable
        delete_result = vector_store.delete([doc1_id])
        assert delete_result is True, "Expected delete to return True"

        fetched_after_delete = vector_store.get([doc1_id])
        assert not fetched_after_delete, "Expected no documents after deletion"
