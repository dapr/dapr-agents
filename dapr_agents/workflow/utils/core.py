import inspect
import logging
from dataclasses import is_dataclass
from pydantic import BaseModel
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


def is_pydantic_model(obj: Any) -> bool:
    """Check if the given type is a subclass of Pydantic's BaseModel."""
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def is_supported_model(cls: Any) -> bool:
    """Checks if a class is a supported message schema (Pydantic, dataclass, or dict)."""
    return cls is dict or is_dataclass(cls) or is_pydantic_model(cls)


def is_valid_routable_model(cls: Any) -> bool:
    return is_dataclass(cls) or is_pydantic_model(cls)


def get_decorated_methods(instance: Any, attribute_name: str) -> Dict[str, Callable]:
    """
    Find all **public** bound methods on `instance` that carry a given decorator attribute.

    This will:
      1. Inspect the class for functions or methods.
      2. Bind them to the instance (so `self` is applied).
      3. Filter in only those where `hasattr(method, attribute_name) is True`.

    Args:
        instance:  Any object whose methods you want to inspect.
        attribute_name:
            The name of the attribute set by your decorator
            (e.g. "_is_task" or "_is_workflow").

    Returns:
        A dict mapping `method_name` → `bound method`.

    Example:
        >>> class A:
        ...     @task
        ...     def foo(self): ...
        ...
        >>> get_decorated_methods(A(), "_is_task")
        {"foo": <bound method A.foo of <A object ...>>}
    """
    discovered: Dict[str, Callable] = {}

    cls = type(instance)
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        # skip private/protected
        if name.startswith("_"):
            continue

        # bind to instance so that signature(self, ...) works
        try:
            bound = getattr(instance, name)
        except Exception as e:
            logger.warning(f"Could not bind method '{name}': {e}")
            continue

        # pick up only those with our decorator flag
        if hasattr(bound, attribute_name):
            discovered[name] = bound
            logger.debug(f"Discovered decorated method: {name}")

    return discovered
