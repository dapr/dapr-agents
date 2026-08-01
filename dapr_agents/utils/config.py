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

"""Configuration helpers for configuration hot-reloading and runtime resolution."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from types import UnionType
from typing import Any, Callable, Union, get_origin, get_args

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigFieldDescriptor:
    """Describes how a configuration key maps to a configuration attribute.

    Attributes:
        target_type: Expected Python type for the coerced value.
        setter: Callable ``(obj, value) -> None`` that applies the value after coercion and validation.
        getter: Optional callable ``() -> Any`` that retrieves the value before coercion.
        validator: Optional idempotent callable ``(value) -> Any`` to validate/transform the coerced value.
        fallback: Value to use if mapping fails or when a default baseline value is needed. Defaults to None.
        sensitive: If ``True``, the value is redacted in log output.
        rebuilds_prompt: If ``True``, the prompt template is rebuilt after update.
        triggers_otel_reload: If ``True``, triggers an OpenTelemetry configuration reload after update.
    """

    target_type: type
    setter: Callable[..., None]
    getter: Callable[[], Any] | None = None
    validator: Callable[..., Any] | None = None
    fallback: Any = None
    sensitive: bool = False
    rebuilds_prompt: bool = False
    triggers_otel_reload: bool = False


def normalize_config_key(key: str) -> str:
    """Default normalization of configuration keys to attribute names."""
    return key.lower().replace("-", "_")


def apply_config_map(target_obj: Any, config_field_map: dict[str, Any]) -> None:
    """
    Apply a map of configuration field names to field descriptors onto a target object.
    """
    for key, descriptor in config_field_map.items():
        try:
            apply_config_update(target_obj=target_obj, key=key, descriptor=descriptor)
        except (ValueError, RuntimeError) as e:
            logger.debug(f"Failed to apply config update for {key}: {e}")


def apply_config_update(
    target_obj: Any,
    *,
    key: str,
    descriptor: ConfigFieldDescriptor,
    value: Any = None,
) -> Any:
    """
    Process and apply a configuration update to an object.
    This function is guaranteed to be idempotent if the processing logic is idempotent.

    Args:
        target_obj: The object to be updated.
        key: The configuration key.
        value: Optional value to process and apply.
            Falls back to the descriptor's getter if not provided (may not be idempotent).
        descriptor: An object describing how to process a value for a particular key.

    Returns:
        The final applied value.

    Raises:
        ValueError: If no value can be retrieved or processing fails.
        RuntimeError: If the value cannot be applied.
    """
    processed_value = process_config_update(key=key, value=value, descriptor=descriptor)

    # Apply via setter callback
    try:
        descriptor.setter(target_obj, processed_value)
    except (AttributeError, TypeError):
        raise RuntimeError(
            f"Could not apply setter for key '{key}' (likely read-only)."
        )

    return processed_value


def process_config_update(
    key: str,
    descriptor: ConfigFieldDescriptor,
    value: Any = None,
) -> Any:
    """
    Process a configuration update by coercing, validating, and transforming a value.
    This function is guaranteed to be idempotent if the processing logic is idempotent.

    Args:
        key: The configuration key.
        value: Optional value to process.
            Falls back to the descriptor's getter if not provided (may not be idempotent).
        descriptor: An object describing how to process a value for a particular key.

    Returns:
        The processed value.

    Raises:
        ValueError: If no value can be retrieved or processing fails.
    """
    if not descriptor:
        raise ValueError(f"Unrecognized config key: {key}.")

    try:
        # Retrieve value using getter callback as a fallback
        if value is None and descriptor.getter:
            try:
                value = descriptor.getter()
            except Exception as e:
                raise ValueError(f"Unable to retrieve value for key '{key}': {e}.")

        # Type coercion
        try:
            if value is None:
                # Pass through unset `None` values
                processed_value = None
            else:
                processed_value = coerce_config_value(value, descriptor.target_type)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for key '{key}': {e}.")

        # Validation/transformation
        if descriptor.validator:
            try:
                processed_value = descriptor.validator(processed_value)
            except Exception as e:
                raise ValueError(f"Validation failed for key '{key}': {e}.")
    except Exception:
        if descriptor.fallback:
            logger.debug(
                f"Using fallback value for key '{key}': {descriptor.fallback!r}"
            )
            return descriptor.fallback
        raise

    return processed_value


def coerce_config_value(value: Any, target_type: type) -> Any:
    """Coerce a configuration value (usually a string) to the target Python type."""
    if isinstance(value, target_type):
        return value

    if target_type is str:
        return str(value)

    if target_type is int:
        return int(float(value))

    if target_type is float:
        return float(value)

    if target_type is bool:
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no"):
                return False
        raise ValueError(f"Cannot coerce {value!r} to bool")

    if target_type is list:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    if target_type is dict:
        if isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError(f"JSON parsed to {type(parsed).__name__}, expected dict")
        if isinstance(value, dict):
            return value
        raise ValueError(f"Cannot coerce {type(value).__name__} to dict")

    # Handle types that are not classes
    origin = get_origin(target_type)
    if origin is not None:
        # Support union and bar syntax
        if origin in (Union, UnionType):
            for arg in get_args(target_type):
                try:
                    return coerce_config_value(value, arg)
                except ValueError:
                    continue
            raise ValueError(f"Cannot coerce {value!r} to any type in {target_type}")

    raise ValueError(f"Unsupported target type: {target_type}")
