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

"""Construction helpers that honour Dapr SDK gRPC inbound size configuration.

These helpers read ``DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES`` from the process
environment (and an optional programmatic override) and, when set, plumb the
value through the underlying SDK's ``max_grpc_message_length`` constructor
argument. This makes raising the gRPC inbound size limit (default 4 MiB)
possible without code changes at every ``DaprClient()`` construction site.

The helper works against the currently released Python SDK, which already
accepts ``max_grpc_message_length``. Once dapr/python-sdk#1023 ships, the SDK
will read the same env var directly; until then, ``dapr-agents`` provides the
read-through path here.

Resolution order (highest precedence first):
    1. Explicit ``max_grpc_message_length`` kwarg passed to
       :func:`dapr_client_kwargs`.
    2. Programmatic override registered via
       :func:`set_inbound_message_size_bytes` (used by ``AgentBase`` during
       initialization).
    3. ``DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES`` environment variable.
    4. SDK / gRPC defaults (4 MiB receive, unlimited send).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from dapr.aio.clients import DaprClient as AsyncDaprClient
from dapr.clients import DaprClient

INBOUND_MESSAGE_SIZE_ENV: str = "DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES"

logger = logging.getLogger(__name__)

_override_inbound_size_bytes: Optional[int] = None


def set_inbound_message_size_bytes(value: Optional[int]) -> None:
    """Register a process-wide override for the gRPC inbound message size.

    Takes precedence over :data:`INBOUND_MESSAGE_SIZE_ENV` and is honoured by
    every subsequent :func:`dapr_client_kwargs` call. Pass ``None`` to clear.

    Intended for callers (e.g. agent initialization) that resolve the limit
    from typed config rather than an env var.

    Args:
        value: Byte count to apply, or ``None`` to clear any previous override.

    Raises:
        ValueError: If ``value`` is a non-positive integer.
    """
    global _override_inbound_size_bytes
    if value is not None and value <= 0:
        raise ValueError(
            f"max_grpc_inbound_message_size_bytes must be a positive integer, got {value!r}"
        )
    _override_inbound_size_bytes = value


def get_inbound_message_size_bytes() -> Optional[int]:
    """Return the current programmatic override, or ``None`` when unset."""
    return _override_inbound_size_bytes


def dapr_client_kwargs(**explicit_kwargs: Any) -> Dict[str, Any]:
    """Return SDK constructor kwargs with the resolved gRPC inbound size applied.

    Honours, in order: explicit ``max_grpc_message_length`` kwarg, programmatic
    override (see :func:`set_inbound_message_size_bytes`), then
    :data:`INBOUND_MESSAGE_SIZE_ENV`. Invalid env values are logged and ignored
    so that construction proceeds with the SDK default (4 MiB).

    Args:
        **explicit_kwargs: Kwargs to pass through to the SDK constructor
            (``http_timeout_seconds``, ``address``, ``interceptors``, ...).

    Returns:
        A new dict suitable for ``DaprClient(**dapr_client_kwargs(...))``.
    """
    resolved = dict(explicit_kwargs)
    if "max_grpc_message_length" in resolved:
        return resolved

    if _override_inbound_size_bytes is not None:
        resolved["max_grpc_message_length"] = _override_inbound_size_bytes
        return resolved

    raw = os.environ.get(INBOUND_MESSAGE_SIZE_ENV)
    if not raw:
        return resolved

    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(
            f"Ignoring invalid {INBOUND_MESSAGE_SIZE_ENV}={raw!r}; "
            "expected an integer byte count"
        )
        return resolved

    if parsed <= 0:
        logger.warning(
            f"Ignoring non-positive {INBOUND_MESSAGE_SIZE_ENV}={raw!r}; "
            "expected a positive integer byte count"
        )
        return resolved

    resolved["max_grpc_message_length"] = parsed
    return resolved


def default_dapr_client_factory() -> DaprClient:
    """Construct a synchronous ``DaprClient`` honouring the resolved size limit."""
    return DaprClient(**dapr_client_kwargs())


def default_async_dapr_client_factory() -> AsyncDaprClient:
    """Construct an asynchronous ``DaprClient`` honouring the resolved size limit."""
    return AsyncDaprClient(**dapr_client_kwargs())
