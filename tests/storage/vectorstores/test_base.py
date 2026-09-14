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

from typing import Any, Dict, List, Optional

from dapr_agents.storage.vectorstores.base import VectorStoreBase
from dapr_agents.types.document import Document


class RecordingVectorStore(VectorStoreBase):
    """Minimal VectorStoreBase implementation that records what add() receives."""

    last_metadatas: Optional[List[Optional[Dict[str, Any]]]] = None

    def add(self, documents, embeddings=None, metadatas=None, **kwargs: Any):
        self.last_metadatas = metadatas
        return [str(i) for i in range(len(list(documents)))]

    def delete(self, ids):
        return True

    def get(self, ids=None):
        return []

    def reset(self):
        pass

    def search_similar(self, query_texts=None, k: int = 4, **kwargs: Any):
        return []


class TestVectorStoreBaseAddDocuments:
    """Test cases for VectorStoreBase.add_documents metadata handling."""

    def test_forwards_metadata_when_only_a_later_document_has_it(self):
        store = RecordingVectorStore()
        documents = [
            Document(text="a", metadata=None),
            Document(text="b", metadata={"source": "file2"}),
        ]

        store.add_documents(documents)

        assert store.last_metadatas == [{}, {"source": "file2"}]

    def test_passes_none_when_no_document_has_metadata(self):
        store = RecordingVectorStore()
        documents = [Document(text="a"), Document(text="b")]

        store.add_documents(documents)

        assert store.last_metadatas is None

    def test_forwards_metadata_when_only_the_first_document_has_it(self):
        store = RecordingVectorStore()
        documents = [
            Document(text="a", metadata={"source": "file1"}),
            Document(text="b", metadata=None),
        ]

        store.add_documents(documents)

        assert store.last_metadatas == [{"source": "file1"}, {}]
