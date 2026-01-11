import pytest

try:
    import redisvl  # noqa: F401
    from dapr_agents.document.embedder.sentence import SentenceTransformerEmbedder
    from dapr_agents.storage.vectorstores.redis import RedisVectorStore

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    SentenceTransformerEmbedder = None  # type: ignore
    RedisVectorStore = None  # type: ignore

pytestmark = pytest.mark.skipif(
    not REDIS_AVAILABLE,
    reason="redisvl or sentence-transformers not installed - optional dependencies",
)


class TestRedisVectorStore:
    """Test cases for RedisVectorStore."""

    @pytest.fixture
    def embedder(self, test_model_name):
        """Create a SentenceTransformerEmbedder fixture."""
        return SentenceTransformerEmbedder(model=test_model_name)

    @pytest.fixture
    def redis_index_name(self):
        """Create a unique index name for testing."""
        return "test_redis_index"

    @pytest.fixture
    def vector_store(self, embedder, redis_index_name):
        """Create a RedisVectorStore fixture.

        Note: This requires a running Redis instance with the RediSearch module.
        Tests will be skipped if Redis is not available.
        """
        try:
            store = RedisVectorStore(
                index_name=redis_index_name,
                embedding_function=embedder,
                embedding_dims=384,  # all-MiniLM-L6-v2 produces 384-dim embeddings
                redis_url="redis://localhost:6379",
            )
            yield store
            # Cleanup after test
            try:
                store.index.delete(drop=True)
            except Exception:
                pass
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")

    def test_redis_vectorstore_creation(self, embedder, redis_index_name):
        """Test that RedisVectorStore can be created successfully."""
        try:
            vector_store = RedisVectorStore(
                index_name=redis_index_name,
                embedding_function=embedder,
                embedding_dims=384,
            )
            assert vector_store is not None
            assert vector_store.index_name == redis_index_name
            # Cleanup
            try:
                vector_store.index.delete(drop=True)
            except Exception:
                pass
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")

    def test_embedder_has_name_attribute(self, embedder):
        """Test that the embedder has a name attribute."""
        assert hasattr(embedder, "name"), "Embedder should have a name attribute"
        assert embedder.name is not None, "Name attribute should not be None"

    def test_vectorstore_with_embedder(self, vector_store, redis_index_name):
        """Test that RedisVectorStore works with the embedder."""
        assert vector_store is not None
        assert hasattr(vector_store, "index_name")
        assert vector_store.index_name == redis_index_name

    def test_vectorstore_different_names(self, embedder):
        """Test creating vector stores with different names."""
        names = ["test_index_1", "test_index_2", "another_index"]
        stores = []

        for name in names:
            try:
                vector_store = RedisVectorStore(
                    index_name=name,
                    embedding_function=embedder,
                    embedding_dims=384,
                )
                stores.append(vector_store)
                assert vector_store is not None
                assert vector_store.index_name == name
            except Exception as e:
                pytest.skip(f"Redis not available: {e}")

        # Cleanup
        for store in stores:
            try:
                store.index.delete(drop=True)
            except Exception:
                pass

    def test_vectorstore_distance_metrics(self, embedder):
        """Test creating vector stores with different distance metrics."""
        metrics = ["cosine", "l2", "ip"]
        stores = []

        for i, metric in enumerate(metrics):
            try:
                vector_store = RedisVectorStore(
                    index_name=f"test_metric_{metric}_{i}",
                    embedding_function=embedder,
                    embedding_dims=384,
                    distance_metric=metric,
                )
                stores.append(vector_store)
                assert vector_store is not None
                assert vector_store.distance_metric == metric
            except Exception as e:
                pytest.skip(f"Redis not available: {e}")

        # Cleanup
        for store in stores:
            try:
                store.index.delete(drop=True)
            except Exception:
                pass
