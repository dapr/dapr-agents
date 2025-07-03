import pytest
from dapr_agents.document.embedder.sentence import SentenceTransformerEmbedder
from dapr_agents.storage.vectorstores.chroma import ChromaVectorStore


class TestChromaVectorStore:
    """Test cases for ChromaVectorStore."""

    @pytest.fixture
    def embedder(self, test_model_name):
        """Create a SentenceTransformerEmbedder fixture."""
        return SentenceTransformerEmbedder(model=test_model_name)

    @pytest.fixture
    def vector_store(self, embedder, test_collection_name):
        """Create a ChromaVectorStore fixture."""
        return ChromaVectorStore(
            name=test_collection_name, embedding_function=embedder, persistent=False
        )

    def test_chroma_vectorstore_creation(self, embedder, test_collection_name):
        """Test that ChromaVectorStore can be created successfully."""
        vector_store = ChromaVectorStore(
            name=test_collection_name, embedding_function=embedder, persistent=False
        )
        assert vector_store is not None
        assert vector_store.name == test_collection_name

    def test_embedder_has_name_attribute(self, embedder):
        """Test that the embedder has a name attribute."""
        assert hasattr(embedder, "name"), "Embedder should have a name attribute"
        assert embedder.name is not None, "Name attribute should not be None"

    def test_vectorstore_with_embedder(self, vector_store, test_collection_name):
        """Test that ChromaVectorStore works with the embedder."""
        assert vector_store is not None
        assert hasattr(vector_store, "name")
        assert vector_store.name == test_collection_name

    def test_vectorstore_persistent_setting(self, embedder):
        """Test that persistent setting is respected."""
        # Test with persistent=False
        vector_store_non_persistent = ChromaVectorStore(
            name="test_collection_non_persistent",
            embedding_function=embedder,
            persistent=False,
        )
        assert vector_store_non_persistent is not None

        # Test with persistent=True
        vector_store_persistent = ChromaVectorStore(
            name="test_collection_persistent",
            embedding_function=embedder,
            persistent=True,
        )
        assert vector_store_persistent is not None

    def test_vectorstore_different_names(self, embedder):
        """Test creating vector stores with different names."""
        names = ["test_collection_1", "test_collection_2", "another_collection"]

        for name in names:
            vector_store = ChromaVectorStore(
                name=name, embedding_function=embedder, persistent=False
            )
            assert vector_store is not None
            assert vector_store.name == name
