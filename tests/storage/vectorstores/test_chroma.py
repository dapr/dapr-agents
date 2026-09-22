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

# ChromaVectorStore only imports the `chromadb` package lazily inside
# model_post_init, so the class itself (and its pure-Python methods, e.g.
# get()) can be imported and unit-tested even when chromadb isn't installed.
from dapr_agents.storage.vectorstores.chroma import ChromaVectorStore

try:
    import chromadb  # noqa: F401
    from dapr_agents.document.embedder.sentence import SentenceTransformerEmbedder

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    SentenceTransformerEmbedder = None

requires_chromadb = pytest.mark.skipif(
    not CHROMA_AVAILABLE,
    reason="chromadb or sentence-transformers not installed - optional dependencies",
)


@requires_chromadb
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


class TestChromaVectorStoreGet:
    """Tests for ChromaVectorStore.get() with a stubbed collection.

    These don't need the real chromadb dependency since get() only touches
    self.collection, which we substitute with a lightweight stub.
    """

    @staticmethod
    def _make_store(collection):
        store = object.__new__(ChromaVectorStore)
        object.__setattr__(store, "collection", collection)
        return store

    def test_get_with_default_include(self):
        class StubCollection:
            def get(self, ids=None, include=None):
                return {
                    "ids": ["1", "2"],
                    "metadatas": [{"a": 1}, {"a": 2}],
                    "documents": ["doc1", "doc2"],
                }

        store = self._make_store(StubCollection())
        result = store.get(ids=["1", "2"])
        assert result == [
            {"id": "1", "metadata": {"a": 1}, "document": "doc1"},
            {"id": "2", "metadata": {"a": 2}, "document": "doc2"},
        ]

    def test_get_with_include_excluding_documents(self):
        # Chroma returns None (not an empty list) for any field left out of
        # `include`; get() must not crash when zipping the results together.
        class StubCollection:
            def get(self, ids=None, include=None):
                return {
                    "ids": ["1", "2"],
                    "metadatas": [{"a": 1}, {"a": 2}],
                    "documents": None,
                }

        store = self._make_store(StubCollection())
        result = store.get(ids=["1", "2"], include=["metadatas"])
        assert result == [
            {"id": "1", "metadata": {"a": 1}, "document": None},
            {"id": "2", "metadata": {"a": 2}, "document": None},
        ]
