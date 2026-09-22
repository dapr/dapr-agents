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

import sys
from types import ModuleType
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from dapr_agents.document.embedder.base import EmbedderBase


class _NoopEmbedder(EmbedderBase):
    def embed(self, query, **kwargs) -> List[Any]:
        return []


@pytest.fixture
def pg_modules(monkeypatch):
    """Install fake psycopg/psycopg_pool/pgvector packages so
    PostgresVectorStore can be constructed without the real drivers
    installed, and hand back the mock cursor for assertions."""
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.description = [("id",), ("document",), ("metadata",), ("similarity",)]
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = [0]

    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn

    fake_pool_module = ModuleType("psycopg_pool")
    fake_pool_module.ConnectionPool = MagicMock(return_value=pool)
    monkeypatch.setitem(sys.modules, "psycopg_pool", fake_pool_module)

    fake_pgvector_psycopg = ModuleType("pgvector.psycopg")
    fake_pgvector_psycopg.register_vector = MagicMock()
    fake_pgvector = ModuleType("pgvector")
    fake_pgvector.psycopg = fake_pgvector_psycopg
    monkeypatch.setitem(sys.modules, "pgvector", fake_pgvector)
    monkeypatch.setitem(sys.modules, "pgvector.psycopg", fake_pgvector_psycopg)

    fake_psycopg_json = ModuleType("psycopg.types.json")
    fake_psycopg_json.Jsonb = lambda value: value
    fake_psycopg_types = ModuleType("psycopg.types")
    fake_psycopg_types.json = fake_psycopg_json
    fake_psycopg = ModuleType("psycopg")
    fake_psycopg.types = fake_psycopg_types
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.types", fake_psycopg_types)
    monkeypatch.setitem(sys.modules, "psycopg.types.json", fake_psycopg_json)

    return cursor


@pytest.fixture
def store(pg_modules):
    from dapr_agents.storage.vectorstores.postgres import PostgresVectorStore

    return PostgresVectorStore(
        connection_string="postgresql://localhost/test",
        embedding_function=_NoopEmbedder(),
    )


class TestSearchSimilarDistanceMetricValidation:
    def test_rejects_unknown_distance_metric(self, store, pg_modules):
        with pytest.raises(ValueError, match="distance_metric"):
            store.search_similar(query_embeddings=[0.1, 0.2], distance_metric="dot")

    @pytest.mark.parametrize("metric", ["cosine", "l2", "inner_product"])
    def test_accepts_known_distance_metrics(self, store, pg_modules, metric):
        store.search_similar(query_embeddings=[0.1, 0.2], distance_metric=metric)
        query = pg_modules.execute.call_args[0][0]
        expected_operator = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}[
            metric
        ]
        assert expected_operator in query


class TestSearchSimilarMetadataFilterBinding:
    def test_non_string_filter_value_is_stringified_for_binding(
        self, store, pg_modules
    ):
        store.search_similar(
            query_embeddings=[0.1, 0.2], metadata_filter={"page": 3, "verified": True}
        )
        params = pg_modules.execute.call_args[0][1]
        assert "page" in params
        assert params[params.index("page") + 1] == "3"
        assert "verified" in params
        assert params[params.index("verified") + 1] == "True"

    def test_embedding_is_bound_as_parameter_not_interpolated(self, store, pg_modules):
        store.search_similar(query_embeddings=[0.1, 0.2])
        query, params = pg_modules.execute.call_args[0]
        assert "ARRAY" not in query
        assert [0.1, 0.2] in params

    def test_limit_param_is_not_shadowed_by_metadata_filter_keys(
        self, store, pg_modules
    ):
        """The k (LIMIT) parameter must survive even when metadata_filter
        is non-empty, since the filter and limit both append to params."""
        store.search_similar(
            query_embeddings=[0.1, 0.2], k=7, metadata_filter={"page": 3}
        )
        params = pg_modules.execute.call_args[0][1]
        assert params[-1] == 7


class TestUpdateFalsyValues:
    def test_empty_metadata_dict_is_still_written(self, store, pg_modules):
        store.update(ids=["a"], metadatas=[{}])
        assert pg_modules.execute.called
        query = pg_modules.execute.call_args[0][0]
        assert "metadata = %s" in query

    def test_empty_document_string_is_still_written(self, store, pg_modules):
        store.update(ids=["a"], documents=[""])
        assert pg_modules.execute.called
        query = pg_modules.execute.call_args[0][0]
        assert "document = %s" in query

    def test_none_values_are_skipped(self, store, pg_modules):
        store.update(ids=["a"], documents=[None], metadatas=[{"k": "v"}])
        query = pg_modules.execute.call_args[0][0]
        assert "document = %s" not in query
        assert "metadata = %s" in query
