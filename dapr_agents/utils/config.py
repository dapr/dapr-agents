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
from decimal import Decimal, InvalidOperation
from types import UnionType
from typing import Any, Callable, Mapping, Union, get_origin, get_args

logger = logging.getLogger(__name__)


def _coerce_integral(value: Any) -> int:
    """Convert a value that represents an exact integer to an ``int``.

    Accepts ints, integral floats (``10.0``), ``Decimal`` values, and numeric
    strings (``"10"``, ``"10.0"``, ``"1e3"``). Conversion is exact: nothing is
    rounded or truncated.

    Raises:
        ValueError: if ``value`` is a ``bool``, is not numeric, is not finite
            (``nan``/``inf``), or has a fractional part (``10.5``).
    """
    if isinstance(value, bool):
        raise ValueError(f"Value must be an integer, got {value!r}")

    try:
        numeric = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Value must be an integer, got {value!r}") from exc

    if not numeric.is_finite() or numeric != numeric.to_integral_value():
        raise ValueError(f"Value must be an integer, got {value!r}")

    return int(numeric)


@dataclass(frozen=True)
class ConfigFieldDescriptor:
    """Describes how a configuration key maps to a configuration attribute.

    Attributes:
        target_type: Expected Python type for the coerced value.
        setter: Callable ``(obj, value) -> None`` that applies the value after coercion and validation.
        getter: Optional callable ``() -> Any`` that retrieves the value before coercion.
            Defaults to ``None``.
        pre_validator: Optional callable ``(value) -> Any`` to validate/transform the raw value before coercion.
            Defaults to ``None`` (no pre-validation).
        validator: Optional idempotent callable ``(value) -> Any`` to validate/transform the coerced value.
            Defaults to ``None`` (no validation).
        raise_on_error: If ``True``, propagates exceptions if mapping fails.
            Otherwise, logs a warning and uses the fallback value.
            Defaults to ``True`` (raise).
        fallback: Default value to apply if mapping fails and ``raise_on_error`` is ``False``.
            Defaults to ``None``.
        sensitive: If ``True``, the value is redacted in log output.
        rebuilds_prompt: (Used for hot-reloading) If ``True``, the prompt template is rebuilt after update.
        triggers_otel_reload: (Used for hot-reloading) If ``True``, triggers an OpenTelemetry configuration reload after update.
    """

    target_type: type
    setter: Callable[..., None]
    getter: Callable[[], Any] | None = None
    pre_validator: Callable[..., Any] | None = None
    validator: Callable[..., Any] | None = None
    raise_on_error: bool = True
    fallback: Any = None
    sensitive: bool = False
    rebuilds_prompt: bool = False
    triggers_otel_reload: bool = False


def get_config_value(mapping: Mapping[str, Any], key: str) -> Any:
    """
    Get a config value from a mapping under common naming conventions
    in the following order (first non-``None``, non-empty value wins):
        1. SCREAMING_SNAKE_CASE
        2. snake_case
        3. kebab-case

    Returns:
        The value for the key, or ``None`` if no variant is present.
    """
    snake = key.replace("-", "_")
    kebab = key.replace("_", "-")
    for candidate in (snake.upper(), snake.lower(), kebab.lower()):
        value = mapping.get(candidate, None)
        if value is not None and value != "":
            return value
    return None


def normalize_config_key(key: str) -> str:
    """
    Normalize a configuration key to snake_case.
    Accepts SCREAMING_SNAKE_CASE, snake_case, and kebab-case naming conventions.

    Returns:
        The normalized key in snake_case.

    Raises:
        ValueError: If the key does not follow a supported naming convention.
    """
    normalized = key.lower().replace("-", "_")
    is_lower = key == key.lower()
    is_screaming_snake = key == key.upper() and "-" not in key
    if not (is_lower or is_screaming_snake):
        raise ValueError(
            f"Configuration key '{key}' does not follow a supported naming convention "
            "(SCREAMING_SNAKE_CASE, snake_case, kebab-case). "
        )
    return normalized


def apply_config_map(
    target_obj: Any, config_field_map: dict[str, ConfigFieldDescriptor]
) -> None:
    """
    Apply a map of configuration field names to field descriptors onto a target object, mutating the object in-place.

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

    The update is applied using the value returned by the descriptor's processing logic.
    If processing or application fails, behavior depends on the descriptor's ``raise_on_error`` field:

    - If ``raise_on_error`` is ``True``, the error is propagated.
    - If ``raise_on_error`` is ``False``, the error is logged and, when a fallback value is configured,
      the fallback is attempted. If the fallback also fails, the error is logged and the update is skipped.

    Callers **should** ensure that setters are atomic — if a setter mutates the target object before failing,
    callers may observe a partial update.

    Args:
        target_obj: The object to be updated.
        key: The configuration key.
        descriptor: An object describing how to process a value for a particular key.
        value: Optional value to process and apply.
            Falls back to the descriptor's getter if not provided.

    Returns:
        The final applied value, or ``None`` if no value can be applied and the descriptor's ``raise_on_error`` is ``False``.

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
        except Exception as exc:
            raise RuntimeError(f"Could not apply setter for key '{key}'") from exc

        return processed_value
    except Exception as exc:
        if descriptor.raise_on_error:
            raise

        if descriptor.fallback is None:
            # Omit tracebacks for sensitive keys; chained coercion/validation errors may contain the raw value
            if descriptor.sensitive:
                logger.warning(
                    f"Ignoring failed config update for key '{key}': "
                    f"{type(exc).__name__}"
                )
            else:
                logger.warning(
                    f"Ignoring failed config update for key '{key}'", exc_info=True
                )
            return None

        safe_fallback = "***" if descriptor.sensitive else descriptor.fallback
        logger.debug(f"Using fallback value for key '{key}': {safe_fallback!r}")

        # Best-effort update: fall back to the configured value if available
        try:
            descriptor.setter(target_obj, descriptor.fallback)
        except Exception as exc:
            # Omit tracebacks for sensitive keys; chained coercion/validation errors may contain the raw value
            if descriptor.sensitive:
                logger.warning(
                    f"Failed to apply fallback for key '{key}', continuing without "
                    f"update: {type(exc).__name__}"
                )
            else:
                logger.warning(
                    f"Failed to apply fallback for key '{key}', continuing without update",
                    exc_info=True,
                )
            return None

        return descriptor.fallback


def process_config_update(
    key: str,
    descriptor: ConfigFieldDescriptor,
    value: Any = None,
) -> Any:
    """
    Process a configuration update by coercing, validating, and transforming a value.

    Args:
        key: The configuration key.
        value: Optional value to process.
            Falls back to the descriptor's getter if not provided.
        descriptor: An object describing how to process a value for a particular key.

    Returns:
        The processed value.

    Raises:
        ValueError: If no value can be retrieved or processing fails.
    """
    # Retrieve value using getter callback as a fallback
    if value is None and descriptor.getter:
        try:
            value = descriptor.getter()
        except Exception as exc:
            raise ValueError(f"Unable to retrieve value for key '{key}'") from exc

    # Type coercion
    try:
        if value is None:
            # Pass through unset ``None``` values
            processed_value = None
        else:
            if descriptor.pre_validator:
                value = descriptor.pre_validator(value)
            processed_value = coerce_config_value(value, descriptor.target_type)
    except Exception as exc:
        raise ValueError(f"Invalid value for key '{key}'") from exc

    # Validation/transformation
    if processed_value is not None and descriptor.validator:
        try:
            processed_value = descriptor.validator(processed_value)
        except Exception as exc:
            raise ValueError(f"Validation failed for key '{key}'") from exc

    return processed_value


def coerce_config_value(value: Any, target_type: type) -> Any:
    """
    Coerce a configuration value to the target Python type.
    Coercion explicitly enforces the requested target type; subtypes are not accepted.

    Args:
        value: The configuration value to coerce.
        target_type: The target Python type for the coerced value.

    Returns:
        The value coerced to ``target_type``.
    """
    origin = get_origin(target_type)

    if origin in (Union, UnionType):
        # Handle PEP 604 / ``typing.Union`` types by trying each branch in order
        for arg in get_args(target_type):
            try:
                return coerce_config_value(value, arg)
            except (ValueError, TypeError):
                continue
        raise ValueError(f"Cannot coerce {value!r} to any type in {target_type}")

    if origin is not None:
        # Unwrap parameterized generics such as ``dict[str, str]`` / ``list[int]``
        # to their runtime container type before ``isinstance``
        target_type = origin

    # Explicit check for ``bool`` with target type ``int`` as it subclasses ``int``
    if target_type is int and isinstance(value, bool):
        raise ValueError(f"Cannot coerce {value!r} to int")

    if isinstance(value, target_type):
        return value

    if target_type is str:
        return str(value)

    if target_type is int:
        return _coerce_integral(value)

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
