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

from dataclasses import is_dataclass
from typing import Any, Callable, Callable

from pydantic import BaseModel


def is_pydantic_model(obj: Any) -> bool:
    """Check if the given object is a subclass of Pydantic's BaseModel."""
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def is_supported_config_model(obj: Any) -> bool:
    """Checks if an object is a supported configuration model (Pydantic, dataclass, or dict)."""
    return obj is dict or is_dataclass(obj) or is_pydantic_model(obj)


def get_model_fields(model: Any) -> Any | None:
    """
    Extract field names from a config model.

    Returns:
        Iterable of field names, or None if unsupported type.
    """
    if type(model) is dict:
        return model.keys()

    if is_dataclass(model):
        from dataclasses import fields as dataclass_fields

        return [f.name for f in dataclass_fields(model)]

    if hasattr(model, "model_validate"):
        # Pydantic v2
        return model.model_fields.keys()

    if hasattr(model, "parse_obj"):
        # Pydantic v1
        return model.__fields__.keys()

    return None


def get_model_factory(model: Any) -> Callable[..., Any] | None:
    """
    Get the factory function for creating instances of a config model.

    Returns:
        Callable that takes a dict and returns an instance.
    """
    if type(model) is dict:
        return dict  # type: ignore

    if is_dataclass(model):
        return lambda vals: type(model)(**vals)

    if hasattr(model, "model_validate"):
        # Pydantic v2
        return lambda vals: type(model).model_validate(vals)

    if hasattr(model, "parse_obj"):
        # Pydantic v1
        return lambda vals: type(model).parse_obj(vals)

    return None
