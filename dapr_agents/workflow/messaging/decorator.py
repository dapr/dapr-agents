import logging
from copy import deepcopy
from typing import Any, Callable, Optional, get_type_hints
from dapr_agents.workflow.messaging.utils import (
    is_valid_routable_model,
    extract_message_models,
)

logger = logging.getLogger(__name__)


def message_router(
    func: Optional[Callable[..., Any]] = None,
    *,
    pubsub: Optional[str] = None,
    topic: Optional[str] = None,
    dead_letter_topic: Optional[str] = None,
    broadcast: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for registering message handlers by inspecting type hints on the 'message' argument.

    This decorator:
    - Extracts the expected message model type from function annotations.
    - Stores metadata for routing messages by message schema instead of `event.type`.
    - Supports broadcast messaging.
    - Supports Union[...] and multiple models.

    Args:
        func (Optional[Callable]): The function to decorate.
        pubsub (Optional[str]): The name of the pub/sub component.
        topic (Optional[str]): The topic name for the handler.
        dead_letter_topic (Optional[str]): Dead-letter topic for failed messages.
        broadcast (bool): If True, the message is broadcast to all agents.

    Returns:
        Callable: The decorated function with additional metadata.
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        is_workflow = hasattr(f, "_is_workflow")
        workflow_name = getattr(f, "_workflow_name", None)

        type_hints = get_type_hints(f)
        raw_hint = type_hints.get("message", None)

        message_models = extract_message_models(raw_hint)

        if not message_models:
            raise ValueError(
                f"Message handler '{f.__name__}' must have a 'message' parameter with a valid type hint."
            )

        for model in message_models:
            if not is_valid_routable_model(model):
                raise TypeError(
                    f"Handler '{f.__name__}' has unsupported message type: {model}"
                )

        logger.debug(
            f"@message_router: '{f.__name__}' => models {[m.__name__ for m in message_models]}"
        )

        # Attach metadata for later registration
        f._is_message_handler = True
        f._message_router_data = deepcopy(
            {
                "pubsub": pubsub,
                "topic": topic,
                "dead_letter_topic": dead_letter_topic
                or (f"{topic}_DEAD" if topic else None),
                "is_broadcast": broadcast,
                "message_schemas": message_models,
                "message_types": [model.__name__ for model in message_models],
            }
        )

        if is_workflow:
            f._is_workflow = True
            f._workflow_name = workflow_name

        return f

    return decorator(func) if func else decorator
