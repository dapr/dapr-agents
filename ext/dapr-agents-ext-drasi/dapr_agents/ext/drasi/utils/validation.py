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

from dataclasses import is_dataclass
from typing import Any, TypeVar

from dapr_agents.ext.drasi.schemas import Op
from dapr_agents.workflow.utils.core import is_supported_model

T = TypeVar("T")


def normalize_to_list(value: T | list[T] | None) -> list[T]:
    """Convert a scalar or a list to a list. If the value is `None`, returns an empty list."""
    if value is None:
        return []

    if isinstance(value, list):
        return list(value)

    # Treat sequences (e.g. strings) as scalar values
    return [value]


def is_supported_operation(value: Any) -> bool:
    """Return `True` if the value is a supported Drasi operation, otherwise `False`."""
    try:
        op = Op(value)
        return op in {Op.i, Op.u, Op.d}
    except (ValueError, TypeError):
        return False


def validate_model(model: type[Any], data: dict) -> Any:
    """
    Validate/coerce data into a supported model instance (dict, dataclass, Pydantic v1/v2).

    Mirrors `dapr_agents.workflow.utils.core.validate_model` but does not log exceptions
    as validation failures are expected.

    Args:
        model (type[Any]):
            The model class to validate against.
        data (dict):
            The data to validate.

    Returns:
        Any: The validated/coerced model instance.

    Raises:
        TypeError: If the model type is unsupported.
        ValueError: If the data cannot be validated against the model.
    """
    if not is_supported_model(model):
        raise TypeError(f"Unsupported model type: {model!r}")

    try:
        if model is dict:
            return data

        if is_dataclass(model):
            return model(**data)

        if hasattr(model, "model_validate"):  # Pydantic v2
            return model.model_validate(data)

        if hasattr(model, "parse_obj"):  # Pydantic v1
            return model.parse_obj(data)

        raise TypeError(f"Unsupported model type: {model!r}")
    except Exception as e:
        raise ValueError(f"Validation failed: {e}") from e
