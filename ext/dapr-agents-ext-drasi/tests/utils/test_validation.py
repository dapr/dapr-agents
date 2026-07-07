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
from dataclasses import dataclass

from pydantic import BaseModel
import pytest

try:
    from dapr_agents.ext.drasi.types import DrasiOperation
    from dapr_agents.ext.drasi.utils.validation import (
        is_supported_operation,
        maybe_coerce_operation,
        normalize_to_list,
        validate_model,
    )

    DRASI_AVAILABLE = True
except ImportError:
    DRASI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DRASI_AVAILABLE,
    reason=(
        "dapr-agents-ext-drasi is not available. "
        "To run these tests, install the extension with: "
        "`uv sync --group test --extra drasi`",
    ),
)


# ---------------------------------------------------------------------------
# normalize_to_list
# ---------------------------------------------------------------------------


def test_normalize_to_list_none_returns_empty_list():
    """Test that `None` is normalized to an empty list."""
    assert normalize_to_list(None) == []


def test_normalize_to_list_scalar_returns_single_element_list():
    """Test that a scalar value is normalized to a single-element list."""
    assert normalize_to_list(1) == [1]


def test_normalize_to_list_scalar_sequence_returns_single_element_list():
    """Test that a scalar sequence is normalized to a single-element list."""
    assert normalize_to_list("hi") == ["hi"]


def test_normalize_to_list_list_returns_copied_list():
    """Test that a list is normalized to a copy of the list."""
    original_list = [{"a": "b"}, 100.0, True]
    result = normalize_to_list(original_list)

    assert result == original_list
    assert result is not original_list


# ---------------------------------------------------------------------------
# is_supported_operation
# ---------------------------------------------------------------------------


def test_is_supported_operation_accepts_valid_enum_values():
    """Test that supported Drasi operations return `True`."""
    assert is_supported_operation(DrasiOperation.i)
    assert is_supported_operation(DrasiOperation.u)
    assert is_supported_operation(DrasiOperation.d)


def test_is_supported_operation_accepts_valid_string_literal_values():
    """Test that string literal equivalents of supported Drasi operations return `True`."""
    assert is_supported_operation("i")
    assert is_supported_operation("u")
    assert is_supported_operation("d")


def test_is_supported_operation_rejects_invalid_values():
    """Test that unsupported Drasi operations and invalid values return `False`."""
    assert not is_supported_operation("x")
    assert not is_supported_operation("h")
    assert not is_supported_operation("r")
    assert not is_supported_operation("invalid")
    assert not is_supported_operation(456)
    assert not is_supported_operation(False)
    assert not is_supported_operation(None)


# ---------------------------------------------------------------------------
# maybe_coerce_operation
# ---------------------------------------------------------------------------


def test_maybe_coerce_operation_coerces_valid_enum_values():
    """Test that supported Drasi operations are normalized."""
    assert maybe_coerce_operation(DrasiOperation.i) == DrasiOperation.i
    assert maybe_coerce_operation(DrasiOperation.u) == DrasiOperation.u
    assert maybe_coerce_operation(DrasiOperation.d) == DrasiOperation.d


def test_maybe_coerce_operation_coerces_valid_string_literal_values():
    """Test that string literal equivalents of supported Drasi operations are normalized."""
    assert maybe_coerce_operation("i") == DrasiOperation.i
    assert maybe_coerce_operation("u") == DrasiOperation.u
    assert maybe_coerce_operation("d") == DrasiOperation.d


def test_maybe_coerce_operation_returns_invalid_values_unchanged():
    """Test that unsupported Drasi operations and invalid values are returned unchanged."""
    assert maybe_coerce_operation("x") == "x"
    assert maybe_coerce_operation("h") == "h"
    assert maybe_coerce_operation("r") == "r"
    assert maybe_coerce_operation("invalid") == "invalid"
    assert maybe_coerce_operation(456) == 456
    assert maybe_coerce_operation(False) is False
    assert maybe_coerce_operation(None) is None


# ---------------------------------------------------------------------------
# validate_model
# ---------------------------------------------------------------------------


def test_validate_model_pydantic():
    """Test validating data against Pydantic model."""

    class LowStockEvent(BaseModel):
        product_id: str
        quantity: int

    event_data = {"product_id": "1", "quantity": 10}
    result = validate_model(LowStockEvent, event_data)

    assert isinstance(result, LowStockEvent)
    assert result.product_id == "1"
    assert result.quantity == 10


def test_validate_model_dataclass():
    """Test validating data against dataclass model."""

    @dataclass
    class PickupScheduled:
        pickup_id: str
        scheduled_time: str
        cancelled: bool

    event_data = {
        "pickup_id": "P123",
        "scheduled_time": "2026-06-29T12:00:00Z",
        "cancelled": False,
    }
    result = validate_model(PickupScheduled, event_data)

    assert isinstance(result, PickupScheduled)
    assert result.pickup_id == "P123"
    assert result.scheduled_time == "2026-06-29T12:00:00Z"
    assert result.cancelled is False


def test_validate_model_dict():
    """Test validating data against dict model (passthrough)."""
    event_data = {"foo": "bar"}
    result = validate_model(dict, event_data)

    assert result == event_data
    assert isinstance(result, dict)


def test_validate_model_validation_error():
    """Test that validation errors raise ValueError."""

    class TicketCreated(BaseModel):
        ticket_id: str
        priority: int

    # Missing 'priority'
    event_data = {"ticket_id": "T1"}

    with pytest.raises(ValueError, match="Validation failed"):
        validate_model(TicketCreated, event_data)


def test_validate_model_unsupported_type():
    """Test that unsupported model types raise TypeError."""

    class UnsupportedModel:
        pass

    with pytest.raises(TypeError, match="Unsupported model type"):
        validate_model(UnsupportedModel, {})
