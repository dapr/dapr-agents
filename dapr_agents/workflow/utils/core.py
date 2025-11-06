import asyncio
import inspect
import logging
import signal
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


async def wait_for_shutdown() -> None:
    """Block until Ctrl+C or SIGTERM is received."""
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _set_stop(*_: object) -> None:
        stop.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _set_stop)
        loop.add_signal_handler(signal.SIGTERM, _set_stop)
    except (NotImplementedError, RuntimeError):
        # Windows fallback where add_signal_handler is unavailable
        signal.signal(signal.SIGINT, lambda *_: _set_stop())
        signal.signal(signal.SIGTERM, lambda *_: _set_stop())

    await stop.wait()


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


def to_payload(model: Any) -> Dict[str, Any]:
    """
    Convert supported message instances to a JSON-serializable dict.

    Supports:
    - dict: returned as a shallow copy
    - dataclass: converted via asdict()
    - Pydantic v2: model_dump(exclude_none=True)
    - Pydantic v1: dict(exclude_none=True)

    Falls back to dict() if possible, otherwise raises.
    """
    if isinstance(model, dict):
        return dict(model)

    if is_dataclass(model):
        return asdict(model)

    # Pydantic v2
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)

    # Pydantic v1
    if hasattr(model, "dict"):
        return model.dict(exclude_none=True)

    try:
        return dict(model)
    except Exception as e:
        raise TypeError(
            f"Unsupported message payload type for serialization: {type(model)!r}"
        ) from e


def coerce_to_model(model_type: Type[Any], value: Any) -> Any:
    """
    Best-effort coercion of `value` into `model_type` where possible.

    Args:
        model_type: Expected model class/type.
        value: Incoming value.

    Returns:
        Any: Value coerced to `model_type` when feasible; original value otherwise.
    """
    if model_type is Any or model_type is dict:
        return value

    try:
        if isinstance(value, model_type):
            return value
    except TypeError:
        # model_type may be typing constructs (e.g., typing.Dict) that break isinstance
        pass

    if hasattr(model_type, "model_validate"):
        return model_type.model_validate(value)

    if hasattr(model_type, "parse_obj"):
        return model_type.parse_obj(value)

    if is_dataclass(model_type):
        try:
            if isinstance(value, model_type):
                return value
        except TypeError:
            pass
        if isinstance(value, dict):
            return model_type(**value)

    return value
