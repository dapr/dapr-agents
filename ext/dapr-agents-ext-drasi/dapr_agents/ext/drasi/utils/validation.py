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

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeVar

T = TypeVar("T")

# Internal operation enum for validation
class _DrasiOperation(StrEnum):
    INSERT = "i"
    UPDATE = "u"
    DELETE = "d"
    CONTROL = "x"


def is_supported_operation(value: Any) -> bool:
    """Validate that the value is a supported Drasi operation."""
    try:
        _DrasiOperation(value)
        return True
    except (ValueError, TypeError):
        return False


def normalize_to_list(value: T | list[T] | tuple[T, ...] | None) -> list[T]:
    """Convert a scalar, list, or tuple to a list. If the value is `None`, returns an empty list."""
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    # Treat strings as scalar values
    return [value]
