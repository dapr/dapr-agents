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

from dapr_agents.document.embedder.sentence import SentenceTransformerEmbedder

# Check if sentence-transformers is available
try:
    import sentence_transformers  # noqa: F401

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not installed",
)


class TestSentenceTransformerEmbedder:
    """Test cases for SentenceTransformerEmbedder."""

    def test_embedder_creation(self, test_model_name):
        """Test that SentenceTransformerEmbedder can be created successfully."""
        embedder = SentenceTransformerEmbedder(model=test_model_name)
        assert embedder is not None

    def test_embedder_has_name_attribute(self, test_model_name):
        """Test that SentenceTransformerEmbedder has a name attribute."""
        embedder = SentenceTransformerEmbedder(model=test_model_name)
        assert hasattr(embedder, "name"), "Embedder should have a name attribute"
        assert embedder.name is not None, "Name attribute should not be None"

    def test_embedder_call_method(self, test_model_name):
        """Test that the embedder can be called to generate embeddings."""
        embedder = SentenceTransformerEmbedder(model=test_model_name)
        test_text = "test text"
        result = embedder(test_text)

        assert isinstance(result, list), "Result should be a list"
        assert len(result) > 0, "Result should not be empty"
        assert all(isinstance(x, float) for x in result), (
            "All elements should be floats"
        )

    def test_embedder_with_different_texts(self, test_model_name):
        """Test embedder with various text inputs."""
        embedder = SentenceTransformerEmbedder(model=test_model_name)

        test_texts = [
            "Hello world",
            "This is a longer sentence with more words.",
            "Special chars: !@#$%^&*()",
        ]

        for text in test_texts:
            result = embedder(text)
            assert isinstance(result, list), f"Result for '{text}' should be a list"
            assert len(result) > 0, f"Result for '{text}' should not be empty"
