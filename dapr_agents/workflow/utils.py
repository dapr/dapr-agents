import inspect
import logging

logger = logging.getLogger(__name__)

def get_callable_decorated_methods(instance, decorator_attr: str) -> dict:
    """
    Safely retrieves all instance methods decorated with a specific attribute (e.g. `_is_task`, `_is_workflow`).

    Args:
        instance: The class instance to inspect.
        decorator_attr (str): The attribute name set by a decorator (e.g. "_is_task").

    Returns:
        dict: Mapping of method names to bound method callables.
    """
    discovered = {}
    for method_name in dir(instance):
        if method_name.startswith("_"):
            continue  # Skip private/protected

        raw_attr = getattr(type(instance), method_name, None)
        if not (inspect.isfunction(raw_attr) or inspect.ismethod(raw_attr)):
            continue  # Skip non-methods (e.g., @property)

        try:
            method = getattr(instance, method_name)
        except Exception as e:
            logger.warning(f"Skipping method '{method_name}' due to error: {e}")
            continue

        if hasattr(method, decorator_attr):
            discovered[method_name] = method

    return discovered