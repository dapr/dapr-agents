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

"""Model helpers for model validation and resolution."""

from __future__ import annotations

import logging

from dataclasses import fields, is_dataclass
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_pydantic_model(obj: Any) -> bool:
    """Checks if the given object is a subclass of Pydantic's BaseModel."""
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def is_supported_model(obj: Any) -> bool:
    """Checks if an object is a supported model (Pydantic, dataclass, or dict)."""
    return obj is dict or is_dataclass(obj) or is_pydantic_model(obj)


def is_supported_model_instance(obj: Any) -> bool:
    """Checks if an object is an instance of a supported model (Pydantic, dataclass, or dict)."""
    return isinstance(obj, dict) or is_dataclass(obj) or isinstance(obj, BaseModel)


def get_model_fields(model: Any) -> Any:
    """Returns field names for a model."""
    if type(model) is dict:
        return model.keys()

    if is_dataclass(model):
        return [f.name for f in fields(model)]

    if hasattr(model, "model_validate"):
        # Pydantic v2
        return model.__class__.model_fields.keys()

    if hasattr(model, "parse_obj"):
        # Pydantic v1
        return model.__fields__.keys()

    raise TypeError(f"Unsupported model type: {model!r}")


def get_model_factory(model: Any) -> Callable[..., Any]:
    """Returns a factory function that takes a dictionary of values and creates a model instance."""
    if type(model) is dict:
        return lambda vals: dict(**vals)

    if is_dataclass(model):
        return lambda vals: type(model)(**vals)  # type: ignore[misc]

    if hasattr(model, "model_validate"):
        # Pydantic v2
        return lambda vals: type(model).model_validate(vals)

    if hasattr(model, "parse_obj"):
        # Pydantic v1
        return lambda vals: type(model).parse_obj(vals)

    raise TypeError(f"Unsupported model type: {model!r}")


def merge_models(base: T, override: T) -> T:
    """
    Merge two models of the same type, with override taking precedence.
    Only override if the override value is not None.
    If merging fails, falls back to the base model if it is valid, otherwise falls back to the override model.

    Args:
        base: The base model.
        override: The new model with potential override values.

    Returns:
        The merged model.
    """
    if not is_supported_model(type(base)):
        logger.warning(f"Unsupported model type: {base!r}")
        return override

    if not is_supported_model(type(override)):
        logger.warning(f"Unsupported model type: {override!r}")
        return base

    if base.__class__ != override.__class__:
        logger.warning(
            f"Cannot merge models of different types: {base!r} and {override!r}"
        )
        return base

    # If both models are dictionaries, perform a direct shallow merge (more performant)
    if isinstance(base, dict) and isinstance(override, dict):
        return {**base, **override}  # type: ignore[return-value]

    try:
        # Infer model type from the base model
        model_fields = get_model_fields(base)
        model_factory = get_model_factory(base)

        logger.debug(
            (f"Merging models:\nBase model: {base!r}\nOverride model: {override!r}")
        )

        merged_fields: dict[str, Any] = {}

        for model_field in model_fields:
            base_field = getattr(base, model_field)
            override_field = getattr(override, model_field)

            if isinstance(base_field, dict) and isinstance(override_field, dict):
                # Shallow merge dicts
                merged_fields[model_field] = {**base_field, **override_field}
            else:
                merged_fields[model_field] = (
                    override_field if override_field is not None else base_field
                )

        model = model_factory(merged_fields)

        logger.debug(f"Merged model: {model!r}")
        return model
    except Exception:
        logger.warning("Failed to merge models", exc_info=True)
        return base
