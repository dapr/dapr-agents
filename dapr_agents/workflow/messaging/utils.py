from pydantic import BaseModel
from dataclasses import is_dataclass
from typing import Any, Union, get_args, get_origin


def is_pydantic_model(cls: Any) -> bool:
    return isinstance(cls, type) and issubclass(cls, BaseModel)


def is_valid_routable_model(cls: Any) -> bool:
    return is_dataclass(cls) or is_pydantic_model(cls)


def is_supported_model(cls: Any) -> bool:
    """Checks if a class is a supported message schema (Pydantic, dataclass, or dict)."""
    return cls is dict or is_dataclass(cls) or is_pydantic_model(cls)


def extract_message_models(type_hint: Any) -> list[type]:
    """
    Extracts one or more message types from a type hint.

    Supports:
    - Single type hint: `MyMessage`
    - Union types: `Union[MessageA, MessageB]`
    - Fallback to empty list if not valid
    """
    if type_hint is None:
        return []

    origin = get_origin(type_hint)
    if origin is Union:
        return list(get_args(type_hint))
    else:
        return [type_hint]
