from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Type


class DaprWorkflowStatus(str, Enum):
    """Enumeration of possible workflow statuses for standardized tracking."""

    UNKNOWN = "unknown"  # Workflow is in an undefined state
    RUNNING = "running"  # Workflow is actively running
    COMPLETED = "completed"  # Workflow has completed
    FAILED = "failed"  # Workflow encountered an error
    TERMINATED = "terminated"  # Workflow was canceled or forcefully terminated
    SUSPENDED = "suspended"  # Workflow was temporarily paused
    PENDING = "pending"  # Workflow is waiting to start


@dataclass
class PubSubRouteSpec:
    """
    Pub/sub subscription that schedules a workflow when a message arrives.

    Attributes:
        pubsub_name: Dapr pub/sub component name.
        topic: Topic to subscribe to.
        handler_fn: Bound workflow callable to run (method or function).
        message_model: Optional schema (Pydantic/dataclass/dict). If omitted and
            `handler_fn` is decorated with `@message_router`, the decorator's
            first schema is used; otherwise `dict`.
        dead_letter_topic: Optional DLQ topic name.
    """

    pubsub_name: str
    topic: str
    handler_fn: Callable[..., Any]
    message_model: Optional[Type[Any]] = None
    dead_letter_topic: Optional[str] = None


@dataclass
class HttpRouteSpec:
    """
    HTTP endpoint that schedules a workflow when a request arrives.

    Attributes:
        path: FastAPI path to mount (e.g., "/blog/start").
        handler_fn: Bound workflow callable to run (method or function).
        method: HTTP method (default: POST).
        request_model: Optional schema for request validation. If omitted and
            `handler_fn` is decorated with `@http_router`, the decorator's first
            schema is used; otherwise `dict`.
        summary: Optional OpenAPI summary.
        tags: Optional OpenAPI tags.
        response_model: Optional Pydantic response model for docs.
    """

    path: str
    handler_fn: Callable[..., Any]
    method: str = "POST"
    request_model: Optional[Type[Any]] = None
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    response_model: Optional[Type[Any]] = None
