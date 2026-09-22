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
from unittest.mock import MagicMock

import pytest

from dapr_agents.types import Node, Relationship


class _FakeNeo4jError(Exception):
    pass


class _FakeCypherSyntaxError(_FakeNeo4jError):
    pass


@pytest.fixture
def neo4j_module(monkeypatch):
    """Install a fake `neo4j` package so Neo4jGraphStore can be constructed
    without the real driver installed."""
    fake_neo4j = ModuleType("neo4j")
    fake_neo4j.GraphDatabase = MagicMock()
    fake_neo4j.Query = lambda text, timeout=None: text

    fake_exceptions = ModuleType("neo4j.exceptions")
    fake_exceptions.Neo4jError = _FakeNeo4jError
    fake_exceptions.CypherSyntaxError = _FakeCypherSyntaxError

    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)
    monkeypatch.setitem(sys.modules, "neo4j.exceptions", fake_exceptions)
    return fake_neo4j


@pytest.fixture
def graph_store(neo4j_module):
    from dapr_agents.storage.graphstores.neo4j.base import Neo4jGraphStore

    return Neo4jGraphStore(uri="bolt://localhost:7687", user="neo4j", password="pw")


class TestAddNodesLabelValidation:
    def test_rejects_label_with_backtick_injection(self, graph_store):
        malicious = Node(
            id="1",
            label="Foo`) DETACH DELETE (n) //",
            properties={"name": "Alice"},
        )
        with pytest.raises(ValueError, match="node label"):
            graph_store.add_nodes([malicious])

    def test_accepts_plain_alnum_label(self, graph_store):
        node = Node(id="1", label="Person", properties={"name": "Alice"})
        graph_store.add_nodes([node])  # should not raise


class TestAddRelationshipsTypeValidation:
    def test_rejects_type_with_injection(self, graph_store):
        malicious = Relationship(
            source_node_id="1",
            target_node_id="2",
            type="FRIEND`}]-(x) DETACH DELETE x //",
        )
        with pytest.raises(ValueError, match="relationship type"):
            graph_store.add_relationships([malicious])

    def test_accepts_plain_alnum_type(self, graph_store):
        rel = Relationship(source_node_id="1", target_node_id="2", type="FRIEND")
        graph_store.add_relationships([rel])  # should not raise


class TestCreateVectorIndexValidation:
    def test_rejects_invalid_label(self, graph_store):
        with pytest.raises(ValueError, match="node label"):
            graph_store.create_vector_index(
                label="Foo`) RETURN 1 //", property="embedding", dimensions=8
            )

    def test_rejects_invalid_property(self, graph_store):
        with pytest.raises(ValueError, match="property"):
            graph_store.create_vector_index(
                label="Chunk", property="a) //", dimensions=8
            )

    def test_rejects_invalid_similarity_function(self, graph_store):
        with pytest.raises(ValueError, match="similarity_function"):
            graph_store.create_vector_index(
                label="Chunk",
                property="embedding",
                dimensions=8,
                similarity_function="cosine', 'vector.quantization.enabled': false} //",
            )

    def test_accepts_valid_arguments(self, graph_store):
        graph_store.create_vector_index(
            label="Chunk", property="embedding", dimensions=8
        )  # should not raise
