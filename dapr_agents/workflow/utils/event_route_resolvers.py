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

"""Turn a validated pub/sub message into the fields a workflow event route raises.

Resolvers (dotted field paths or callables), identifier coercion, reserved
event names and payload serialization. Every failure is an
:class:`EventRouteResolutionError`, which drops the message: retrying the same
message gives the same result.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Callable

from dapr_agents.streaming.keys import (
    APPROVAL_RESPONSE_EVENT_PREFIX,
    USER_INPUT_EVENT_PREFIX,
)
from dapr_agents.types.workflow import FieldResolver

if TYPE_CHECKING:
    from dapr_agents.workflow.utils.subscription import MessageContext

# Longest resolved instance id / event name accepted from a message, in
# characters (not bytes).
MAX_EVENT_IDENTIFIER_LENGTH = 512

# Event names the SDK itself waits on, case-folded because the durabletask
# worker matches event names with ``str.casefold()``.
RESERVED_EVENT_NAME_PREFIXES: tuple[str, ...] = (
    APPROVAL_RESPONSE_EVENT_PREFIX.casefold(),
    f"{USER_INPUT_EVENT_PREFIX}:".casefold(),
)


class EventRouteResolutionError(Exception):
    """A message could not be turned into (instance_id, event_name, data).

    Attributes:
        field: The field that failed (``instance_id``, ``event_name``,
            ``data``, or ``message`` for an unexpected error).
        reason: Human-readable reason.
    """

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


def is_reserved_event_name(name: str) -> bool:
    """True when ``name`` starts with an event-name prefix the SDK waits on.

    Compared case-folded, the way the durabletask worker matches event names,
    so ``"approval_re\u017fponse_x"`` (long s) is reserved too.
    """
    return name.casefold().startswith(RESERVED_EVENT_NAME_PREFIXES)


def parse_field_path(path: str) -> tuple[str, ...]:
    """Split and validate a dotted field path.

    Raises:
        ValueError: If the path is empty, has empty or padded segments, or a
            segment starts with ``__``.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    segments = tuple(path.split("."))
    for seg in segments:
        if not seg:
            raise ValueError("empty segment")
        if seg != seg.strip():
            raise ValueError(f"segment {seg!r} has surrounding whitespace")
        if seg.startswith("__"):
            raise ValueError(f"segment {seg!r} may not start with '__'")
    return segments


def _step(cur: Any, seg: str, path: str, field: str) -> Any:
    if isinstance(cur, Mapping):
        if seg in cur:
            return cur[seg]
    elif isinstance(cur, (list, tuple)):
        # Sequences only take numeric indexes (no `items.count` method lookups).
        if seg.isdigit() and int(seg) < len(cur):
            return cur[int(seg)]
    elif hasattr(cur, seg):
        return getattr(cur, seg)
    raise EventRouteResolutionError(field, f"path {path!r}: segment {seg!r} not found")


def resolve_field(
    resolver: FieldResolver,
    message: Any,
    msg_ctx: MessageContext,
    *,
    field: str,
) -> Any:
    """Resolve one field from the validated message.

    Raises:
        EventRouteResolutionError: If the path does not resolve or the callable raises.
    """
    if isinstance(resolver, str):
        try:
            segments = parse_field_path(resolver)
        except ValueError as exc:
            raise EventRouteResolutionError(
                field, f"invalid field path {resolver!r}: {exc}"
            ) from exc
        cur = message
        for seg in segments:
            cur = _step(cur, seg, resolver, field)
        return cur
    try:
        return resolver(message, msg_ctx)
    except Exception as exc:
        raise EventRouteResolutionError(
            field, f"resolver raised {type(exc).__name__}: {exc}"
        ) from exc


def coerce_identifier(value: Any, *, field: str) -> str:
    """Coerce a resolved instance id / event name to a non-empty string.

    Raises:
        EventRouteResolutionError: If the value is None, empty, not str/int, or
            longer than ``MAX_EVENT_IDENTIFIER_LENGTH``.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if isinstance(value, str):
        if not value.strip():
            raise EventRouteResolutionError(field, "resolved to an empty string")
        if len(value) > MAX_EVENT_IDENTIFIER_LENGTH:
            raise EventRouteResolutionError(
                field,
                f"resolved to {len(value)} characters "
                f"(limit {MAX_EVENT_IDENTIFIER_LENGTH})",
            )
        return value
    if value is None:
        raise EventRouteResolutionError(field, "resolved to None")
    raise EventRouteResolutionError(
        field, f"resolved to {type(value).__name__}, expected str"
    )


def serialize_event_data(
    resolver: FieldResolver | None,
    message: Any,
    msg_ctx: MessageContext,
    *,
    default_serializer: Callable[[Any], Any],
    max_bytes: int | None = None,
) -> Any:
    """Build the JSON-safe event payload.

    Raises:
        EventRouteResolutionError: If resolution or serialization fails in any
            way (including recursion errors), the result is not
            JSON-serializable, or it is larger than ``max_bytes`` once encoded.
            A poison message must be dropped, never retried.
    """
    if resolver is None:
        try:
            result = default_serializer(message)
        except EventRouteResolutionError:
            raise
        except Exception as exc:
            raise EventRouteResolutionError(
                "data", f"serialization failed: {type(exc).__name__}: {exc}"
            ) from exc
    else:
        result = resolve_field(resolver, message, msg_ctx, field="data")
    try:
        result = _to_json_value(result)
        encoded = json.dumps(result)
    except Exception as exc:
        raise EventRouteResolutionError(
            "data", f"not JSON-serializable: {type(exc).__name__}: {exc}"
        ) from exc
    size = len(encoded.encode("utf-8"))
    if max_bytes is not None and size > max_bytes:
        raise EventRouteResolutionError(
            "data", f"serialized payload is {size} bytes (limit {max_bytes})"
        )
    return result


def _to_json_value(value: Any) -> Any:
    """Dump a resolved Pydantic model / dataclass; other values pass through."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


__all__ = [
    "MAX_EVENT_IDENTIFIER_LENGTH",
    "RESERVED_EVENT_NAME_PREFIXES",
    "EventRouteResolutionError",
    "coerce_identifier",
    "is_reserved_event_name",
    "parse_field_path",
    "resolve_field",
    "serialize_event_data",
]
