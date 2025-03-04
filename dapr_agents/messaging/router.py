from typing import Callable, Any, Optional, get_type_hints
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

def message_router(
    func: Optional[Callable[..., Any]] = None,
    *,
    pubsub: Optional[str] = None,
    topic: Optional[str] = None,
    route: Optional[str] = None,
    dead_letter_topic: Optional[str] = None,
    broadcast: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for dynamically registering message handlers for pub/sub topics.

    If applied to a workflow, it marks the function with `_is_workflow` and 
    `_workflow_name`, allowing `register_message_routes()` to process it correctly.

    If applied to a regular function, it stores metadata for routing messages.

    Args:
        func (Optional[Callable]): The function to decorate.
        pubsub (Optional[str]): The name of the pub/sub component.
        topic (Optional[str]): The topic name for the handler.
        route (Optional[str]): The custom route for this handler.
        dead_letter_topic (Optional[str]): Name of the dead letter topic for failed messages.
        broadcast (bool): Indicates if the message should be broadcast to multiple subscribers.

    Returns:
        Callable: The decorated function with additional metadata.
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        is_workflow = hasattr(f, "_is_workflow")
        workflow_name = getattr(f, "_workflow_name", None)

        # Attach routing metadata
        f._is_message_handler = True
        f._message_router_data = deepcopy({
            "pubsub": pubsub,
            "topic": topic,
            "route": route,
            "dead_letter_topic": dead_letter_topic,
            "is_broadcast": broadcast,
            "message_model": get_type_hints(f).get("message"),
        })

        # Preserve workflow metadata if applicable
        if is_workflow:
            f._is_workflow = True
            f._workflow_name = workflow_name

        return f

    return decorator(func) if func else decorator