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

from unittest.mock import MagicMock, patch

import numpy as np

from dapr_agents.document.embedder.openai import OpenAIEmbedder


def _embedding_response(vectors):
    """Build a mock CreateEmbeddingResponse exposing one .embedding per vector."""
    response = MagicMock()
    response.data = [MagicMock(embedding=list(v)) for v in vectors]
    return response


class TestOpenAIEmbedder:
    """Unit tests for OpenAIEmbedder.embed normalization (no network)."""

    def test_single_input_is_normalized_by_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        embedder = OpenAIEmbedder()
        with patch.object(
            OpenAIEmbedder,
            "create_embedding",
            return_value=_embedding_response([[3.0, 4.0]]),
        ):
            result = embedder.embed("hello")
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_list_input_is_normalized_by_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        embedder = OpenAIEmbedder()
        with patch.object(
            OpenAIEmbedder,
            "create_embedding",
            return_value=_embedding_response([[3.0, 4.0], [6.0, 8.0]]),
        ):
            results = embedder.embed(["hello", "world"])
        assert len(results) == 2
        for vector in results:
            assert np.isclose(np.linalg.norm(vector), 1.0)

    def test_normalize_false_returns_raw_vector(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        embedder = OpenAIEmbedder(normalize=False)
        with patch.object(
            OpenAIEmbedder,
            "create_embedding",
            return_value=_embedding_response([[3.0, 4.0]]),
        ):
            result = embedder.embed("hello")
        assert result == [3.0, 4.0]
        assert np.isclose(np.linalg.norm(result), 5.0)
