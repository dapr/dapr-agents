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
environment (and an optional :class:`DaprClientConfig` passed by the caller)
and, when set, plumb the value through the underlying SDK's
``max_grpc_message_length`` constructor argument. This makes raising the gRPC
inbound size limit (default 4 MiB) possible without code changes at every
``DaprClient()`` construction site.

The helper works against the currently released Python SDK, which already
accepts ``max_grpc_message_length``. Once dapr/python-sdk#1023 ships, the SDK
will read the same env var directly; until then, ``dapr-agents`` provides the
read-through path here.

Resolution order (highest precedence first):
    1. Explicit ``max_grpc_message_length`` kwarg passed to
       :func:`dapr_client_kwargs`.
    2. ``config.max_grpc_message_length`` from a :class:`DaprClientConfig`
       passed by the caller (used by ``AgentBase`` to build a single client
       factory shared by all internal Dapr client constructions).
    3. ``DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES`` environment variable.
    4. SDK / gRPC defaults (4 MiB receive, unlimited send).

Components inside ``dapr-agents`` (storage, memory, LLM, runners) accept a
``client_factory`` of type :data:`DaprClientFactory` /
:data:`AsyncDaprClientFactory` and call it whenever they need a fresh client,
so the agent only carries one factory rather than N copies of a config to keep
in sync.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from dapr.aio.clients import DaprClient as AsyncDaprClient
from dapr.clients import DaprClient

INBOUND_MESSAGE_SIZE_ENV: str = "DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES"

logger = logging.getLogger(__name__)

DaprClientFactory = Callable[[], DaprClient]
AsyncDaprClientFactory = Callable[[], AsyncDaprClient]


def _coerce_integral(value: Any) -> int:
    """Convert an integral numeric representation to an ``int``."""
    if isinstance(value, bool):
        raise ValueError(f"Value must be an integer, got {value!r}")

    try:
        numeric = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Value must be an integer, got {value!r}") from exc

    if not numeric.is_finite() or numeric != numeric.to_integral_value():
        raise ValueError(f"Value must be an integer, got {value!r}")

    return int(numeric)


def _coerce_positive_int(value: Any) -> int:
    """Convert a positive integral numeric representation to an ``int``."""
    parsed = _coerce_integral(value)
    if parsed <= 0:
        raise ValueError(
            f"max_grpc_message_length must be a positive integer, got {value!r}"
        )
    return parsed


@dataclass(frozen=True)
class DaprClientConfig:
    """Immutable Dapr client configuration carried explicitly between layers.

    Threading this config through a single client factory replaces process-wide
    module state, so two agents in the same process can run with independent
    limits and tests do not need to reset global state.

    Attributes:
        max_grpc_message_length: gRPC inbound message size limit in bytes.
            Integral numeric values such as ``3.0`` and ``"3.0"`` are normalized
            to ``int``. When ``None``, the env var (and ultimately the SDK
            default) is used.
    """

    max_grpc_message_length: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_grpc_message_length is not None:
            object.__setattr__(
                self,
                "max_grpc_message_length",
                _coerce_positive_int(self.max_grpc_message_length),
            )


def dapr_client_kwargs(
    config: Optional[DaprClientConfig] = None,
    **explicit_kwargs: Any,
) -> Dict[str, Any]:
    """Return SDK constructor kwargs with the resolved gRPC inbound size applied.

    Honours, in order: explicit ``max_grpc_message_length`` kwarg, the value on
    ``config`` (if provided), then :data:`INBOUND_MESSAGE_SIZE_ENV`. Invalid or
    non-positive env values are logged and ignored so construction proceeds
    with the SDK default (4 MiB).

    Args:
        config: Optional typed config carrying ``max_grpc_message_length``.
        **explicit_kwargs: Kwargs to pass through to the SDK constructor
            (``http_timeout_seconds``, ``address``, ``interceptors``, ...).
            An explicit ``max_grpc_message_length`` of ``None`` is dropped (so
            callers can pass it unconditionally); non-positive or fractional
            values raise ``ValueError``.

    Returns:
        A new dict suitable for ``DaprClient(**dapr_client_kwargs(...))``.

    Raises:
        ValueError: If an explicit ``max_grpc_message_length`` is non-positive
            or has a fractional part.
    """
    resolved = dict(explicit_kwargs)
    if "max_grpc_message_length" in resolved:
        explicit = resolved["max_grpc_message_length"]
        if explicit is None:
            # Drop the unset kwarg so we fall through to config/env resolution.
            del resolved["max_grpc_message_length"]
        else:
            resolved["max_grpc_message_length"] = _coerce_positive_int(explicit)
            return resolved

    if config is not None and config.max_grpc_message_length is not None:
        resolved["max_grpc_message_length"] = config.max_grpc_message_length
        return resolved

    raw = os.environ.get(INBOUND_MESSAGE_SIZE_ENV)
    if not raw:
        return resolved

    try:
        parsed = _coerce_integral(raw)
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
    """Construct a synchronous ``DaprClient`` honouring env-var configuration only."""
    return DaprClient(**dapr_client_kwargs())


def default_async_dapr_client_factory() -> AsyncDaprClient:
    """Construct an asynchronous ``DaprClient`` honouring env-var configuration only."""
    return AsyncDaprClient(**dapr_client_kwargs())


def make_dapr_client_factory(
    config: Optional[DaprClientConfig] = None,
) -> DaprClientFactory:
    """Return a no-arg factory bound to ``config`` for synchronous clients."""

    def _factory() -> DaprClient:
        return DaprClient(**dapr_client_kwargs(config=config))

    return _factory


def make_async_dapr_client_factory(
    config: Optional[DaprClientConfig] = None,
) -> AsyncDaprClientFactory:
    """Return a no-arg factory bound to ``config`` for asynchronous clients."""

    def _factory() -> AsyncDaprClient:
        return AsyncDaprClient(**dapr_client_kwargs(config=config))

    return _factory
