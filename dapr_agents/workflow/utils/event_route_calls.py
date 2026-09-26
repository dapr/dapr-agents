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

"""Helpers for the calls a workflow event route makes.

* Classifying sidecar gRPC errors: "instance not found" and permanent errors
  (never retried).
* Running user hooks (``authorize`` and the event-route filters) with a
  deadline and a strict ``True`` contract.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import grpc

from dapr_agents.workflow.utils.call_deadline import (
    DeadlineCaller,
    DeadlineCallerBusyError,
)

logger = logging.getLogger(__name__)

# gRPC codes that will not change on a redelivery of the same message. They
# are handled like a terminal workflow (dead-letter or drop), never retried.
PERMANENT_SIDECAR_ERROR_CODES: frozenset[grpc.StatusCode] = frozenset(
    {
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.PERMISSION_DENIED,
        grpc.StatusCode.UNAUTHENTICATED,
        grpc.StatusCode.UNIMPLEMENTED,
        grpc.StatusCode.OUT_OF_RANGE,
        grpc.StatusCode.FAILED_PRECONDITION,
    }
)

_NOT_FOUND_DETAILS = "no such instance exists"


def safe_grpc_code(exc: BaseException) -> Any:
    """The status code of a gRPC error; None when ``exc`` is not one or ``code()`` raises."""
    if not isinstance(exc, grpc.RpcError):
        return None
    try:
        return exc.code()  # type: ignore[attr-defined]
    except Exception:
        return None


def _safe_grpc_details(exc: BaseException) -> str | None:
    try:
        details = exc.details()  # type: ignore[attr-defined]
    except Exception:
        return None
    return details if isinstance(details, str) else None


def is_instance_not_found_error(exc: BaseException) -> bool:
    """True when ``exc`` is a gRPC error saying the workflow instance does not exist."""
    code = safe_grpc_code(exc)
    if code is None:
        return False
    if code == grpc.StatusCode.NOT_FOUND:
        return True
    details = _safe_grpc_details(exc)
    return details is not None and _NOT_FOUND_DETAILS in details


def permanent_error_code(exc: BaseException) -> grpc.StatusCode | None:
    """The gRPC status code of ``exc`` when it is a permanent sidecar error, else None."""
    code = safe_grpc_code(exc)
    return code if code in PERMANENT_SIDECAR_ERROR_CODES else None


def run_strict_hook(
    caller: DeadlineCaller,
    hook: Callable[..., Any],
    timeout: float,
    *args: Any,
    kind: str,
    route_name: str,
) -> bool:
    """Run ``hook(*args)`` with a deadline; True only for an exact ``True``.

    Fails closed: a timeout, an exception or any other return value
    (``"False"``, ``1``, a mock) is a rejection, even when the cause was
    transient. The arguments are never logged.
    """
    try:
        result = caller.call(hook, timeout, *args)
    except DeadlineCallerBusyError:
        logger.warning(
            "Event route %r: the hook pool is saturated, %s did not run; "
            "rejecting the message.",
            route_name,
            kind,
        )
        return False
    except TimeoutError:
        logger.warning(
            "Event route %r: %s timed out after %gs; rejecting the message.",
            route_name,
            kind,
            timeout,
        )
        return False
    except Exception as exc:
        logger.warning(
            "Event route %r: %s raised %s; rejecting the message.",
            route_name,
            kind,
            type(exc).__name__,
        )
        return False
    if result is True:
        return True
    if result is not False:
        logger.warning(
            "Event route %r: %s returned %s, not a bool; rejecting the message.",
            route_name,
            kind,
            type(result).__name__,
        )
    return False


__all__ = [
    "PERMANENT_SIDECAR_ERROR_CODES",
    "is_instance_not_found_error",
    "permanent_error_code",
    "run_strict_hook",
    "safe_grpc_code",
]
