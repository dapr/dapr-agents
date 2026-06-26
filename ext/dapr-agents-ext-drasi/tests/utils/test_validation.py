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

"""Unit tests for Drasi validation helpers."""

from __future__ import annotations

from dapr_agents.ext.drasi.utils.validation import (
    is_supported_operation,
    normalize_to_list,
)


# ---------------------------------------------------------------------------
# is_change_operation
# ---------------------------------------------------------------------------


def test_is_change_operation_accepts_valid_values():
    assert is_supported_operation("i")
    assert is_supported_operation("u")
    assert is_supported_operation("d")


def test_is_change_operation_rejects_invalid_values():
    assert not is_supported_operation("x")
    assert not is_supported_operation("h")
    assert not is_supported_operation("r")
    assert not is_supported_operation("invalid")
    assert not is_supported_operation(456)
    assert not is_supported_operation(False)
    assert not is_supported_operation(None)


# ---------------------------------------------------------------------------
# normalize_to_list
# ---------------------------------------------------------------------------


def test_normalize_to_list_none_returns_empty_list():
    assert normalize_to_list(None) == []


def test_normalize_to_list_scalar_returns_single_element_list():
    assert normalize_to_list({"a": "b"}) == [{"a": "b"}]


def test_normalize_to_list_scalar_sequence_returns_single_element_list():
    assert normalize_to_list("hi") == ["hi"]


def test_normalize_to_list_list_returns_ordered_list():
    assert normalize_to_list([1, 2, 3]) == [1, 2, 3]
