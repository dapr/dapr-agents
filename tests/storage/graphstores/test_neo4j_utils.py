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

from dapr_agents.storage.graphstores.neo4j.utils import (
    LIST_LIMIT,
    validate_cypher_identifier,
    value_sanitize,
)


class TestValueSanitize:
    """Test cases for value_sanitize."""

    def test_oversized_list_is_truncated_not_dropped(self):
        """Lists longer than LIST_LIMIT are truncated to their first LIST_LIMIT
        elements rather than dropped, as documented in the Returns docstring."""
        data = list(range(LIST_LIMIT + 50))

        result = value_sanitize(data)

        assert result == list(range(LIST_LIMIT))
        assert len(result) == LIST_LIMIT

    def test_list_at_limit_is_preserved(self):
        """A list exactly at LIST_LIMIT is not truncated."""
        data = list(range(LIST_LIMIT))

        assert value_sanitize(data) == data

    def test_small_list_is_preserved(self):
        """A list within the size limit is returned unchanged."""
        assert value_sanitize([1, 2, 3]) == [1, 2, 3]

    def test_unsupported_type_returns_none(self):
        """Unsupported types are excluded by returning None."""
        assert value_sanitize(object()) is None

    def test_dict_drops_none_valued_keys(self):
        """Keys whose sanitized value is None are dropped, while metadata keys
        prefixed with an underscore are preserved as-is."""
        result = value_sanitize({"_id": 1, "keep": "value", "drop": object()})

        assert result == {"_id": 1, "keep": "value"}


class TestValidateCypherIdentifier:
    """Test cases for validate_cypher_identifier."""

    @pytest.mark.parametrize("name", ["Person", "_private", "Node123", "a"])
    def test_accepts_plain_identifiers(self, name):
        assert validate_cypher_identifier(name, "label") == name

    @pytest.mark.parametrize(
        "name",
        [
            "Foo`) DETACH DELETE (n) //",
            "Foo Bar",
            "Foo-Bar",
            "1Foo",
            "",
        ],
    )
    def test_rejects_unsafe_identifiers(self, name):
        with pytest.raises(ValueError, match="label"):
            validate_cypher_identifier(name, "label")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            validate_cypher_identifier(None, "label")
