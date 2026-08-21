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

from enum import Enum
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
        should_raise: If ``True``, raises an exception on failure to process or apply a value.
            If ``False``, logs a warning and uses the fallback value.
            Defaults to ``True`` (raise).
        fallback: Value to use if mapping fails or when a default baseline value is needed. Defaults to None.
        sensitive: If ``True``, the value is redacted in log output.
        rebuilds_prompt: If ``True``, the prompt template is rebuilt after update.
        triggers_otel_reload: If ``True``, triggers an OpenTelemetry configuration reload after update.
    """

    target_type: type
    setter: Callable[..., None]
    getter: Callable[[], Any] | None = None
    validator: Callable[..., Any] | None = None
    should_raise: bool = True
    fallback: Any = None
    sensitive: bool = False
    rebuilds_prompt: bool = False
    triggers_otel_reload: bool = False


def normalize_config_key(key: str) -> str:
    """Default normalization of configuration keys to attribute names."""
    return key.lower().replace("-", "_")


def apply_config_map(
    target_obj: Any, config_field_map: dict[str, ConfigFieldDescriptor]
) -> None:
    """
    Apply a map of configuration field names to field descriptors onto a target object.

    Raises:
        ValueError: If a config key is unrecognized or processing fails.
        RuntimeError: If a value cannot be applied to the target object.
    """
    for key, descriptor in config_field_map.items():
        apply_config_update(target_obj=target_obj, key=key, descriptor=descriptor)


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
    try:
        processed_value = process_config_update(
            key=key, value=value, descriptor=descriptor
        )

        # Apply via setter callback
        try:
            descriptor.setter(target_obj, processed_value)
        except (AttributeError, TypeError):
            raise RuntimeError(
                f"Could not apply setter for key '{key}' (likely read-only)."
            )
        return processed_value
    except (ValueError, RuntimeError, AttributeError, TypeError) as exc:
        if descriptor.should_raise:
            raise

        if descriptor.fallback is None:
            logger.debug(f"Ignoring failed config update for key '{key}': {exc}")
            return None

        logger.debug(f"Using fallback value for key '{key}': {descriptor.fallback!r}")
        try:
            descriptor.setter(target_obj, descriptor.fallback)
        except (AttributeError, TypeError, RuntimeError):
            logger.debug(
                f"Failed to apply fallback for key '{key}', continuing without update."
            )
        return descriptor.fallback


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
    if processed_value is not None and descriptor.validator:
        try:
            processed_value = descriptor.validator(processed_value)
        except Exception as e:
            raise ValueError(f"Validation failed for key '{key}': {e}.")

    return processed_value


def coerce_config_value(value: Any, target_type: type) -> Any:
    """Coerce a configuration value (usually a string) to the target Python type."""
    origin = get_origin(target_type)

    if origin in (Union, UnionType):
        # Handle PEP 604 / typing.Union types by trying each branch in order
        for arg in get_args(target_type):
            try:
                return coerce_config_value(value, arg)
            except (ValueError, TypeError):
                continue
        raise ValueError(f"Cannot coerce {value!r} to any type in {target_type}")

    if origin is not None:
        # Unwrap parameterized generics such as dict[str, str] / list[int]
        # to their runtime container type before `isinstance`
        target_type = origin

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

    if isinstance(target_type, type) and issubclass(target_type, Enum):
        try:
            return target_type(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot coerce {value!r} to {target_type.__name__}"
            ) from exc

    raise ValueError(f"Unsupported target type: {target_type}")
