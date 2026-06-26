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

from typing import Any, TypeVar

from dapr_agents.ext.drasi.schemas import Op

T = TypeVar("T")


def is_supported_operation(value: Any) -> bool:
    """Return `True` if the value is a supported Drasi operation, otherwise `False`."""
    try:
        op = Op(value)
        return op in {Op.i, Op.u, Op.d}
    except (ValueError, TypeError):
        return False


def normalize_to_list(value: T | list[T] | None) -> list[T]:
    """Convert a scalar or a list to a list. If the value is `None`, returns an empty list."""
    if value is None:
        return []

    if isinstance(value, list):
        return list(value)

    # Treat sequences (e.g. strings) as scalar values
    return [value]
