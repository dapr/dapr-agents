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

from __future__ import annotations

import logging
from threading import Lock
from typing import Callable, Literal, TypeAlias

from dapr_agents.agents.durable import DurableAgent
from dapr_agents.types.activation import ActivationCallback

logger = logging.getLogger(__name__)

DrasiMode: TypeAlias = Literal["static", "dynamic"]

_MODE_ATTRIBUTE = "_dapr_agents_ext_drasi_mode"
_MODE_LOCK = Lock()


_DYNAMIC_APPLICATION_OWNERS: dict[str, object] = {}


class DrasiModeConflictError(ValueError):
    """An agent already has an incompatible Drasi registration."""


class DrasiApplicationConflictError(ValueError):
    """Another dynamic lifecycle is active for this Dapr application."""


def _check_mode_locked(agent: DurableAgent, mode: DrasiMode) -> None:
    previous_mode = getattr(agent, _MODE_ATTRIBUTE, None)
    if previous_mode is not None and (previous_mode != mode or mode == "dynamic"):
        message = (
            f"Cannot register Drasi {mode} mode for agent {agent.name!r}: "
            f"{previous_mode} mode is already registered."
        )
        logger.error("%s", message)
        raise DrasiModeConflictError(message)


def claim_dynamic_mode(agent: DurableAgent) -> None:
    """Reserve dynamic mode on a fresh agent without registering an activation."""
    with _MODE_LOCK:
        _check_mode_locked(agent, "dynamic")
        setattr(agent, _MODE_ATTRIBUTE, "dynamic")


def validate_dynamic_mode(agent: DurableAgent) -> None:
    """Check dynamic mode availability without mutating the agent."""
    with _MODE_LOCK:
        _check_mode_locked(agent, "dynamic")


def validate_claimed_dynamic_mode(agent: DurableAgent) -> None:
    """Confirm this agent still holds the dynamic-mode registration."""
    with _MODE_LOCK:
        if getattr(agent, _MODE_ATTRIBUTE, None) != "dynamic":
            raise DrasiModeConflictError(
                f"Agent {agent.name!r} no longer owns Drasi dynamic mode."
            )


def claim_dynamic_application(agent: DurableAgent) -> Callable[[], None]:
    """Claim one active dynamic lifecycle for an application in this process."""
    app_id = agent.appid
    if not isinstance(app_id, str) or not app_id:
        raise ValueError("Dynamic Drasi subscriptions require a Dapr application ID.")
    token = object()
    with _MODE_LOCK:
        owner = _DYNAMIC_APPLICATION_OWNERS.get(app_id)
        if owner is not None:
            message = (
                f"Dapr application {app_id!r} already has an active dynamic Drasi "
                "lifecycle. Use one logical agent per application."
            )
            logger.error("%s", message)
            raise DrasiApplicationConflictError(message)
        _DYNAMIC_APPLICATION_OWNERS[app_id] = token

    def release() -> None:
        with _MODE_LOCK:
            if _DYNAMIC_APPLICATION_OWNERS.get(app_id) is token:
                del _DYNAMIC_APPLICATION_OWNERS[app_id]

    return release


def register_activation(
    agent: DurableAgent,
    *,
    mode: DrasiMode,
    callback: ActivationCallback,
) -> None:
    """Register an activation without mixing Drasi modes on an agent.

    Mode ownership lasts as long as the registered callbacks, including across
    failed hosting attempts and subscription shutdown.
    """
    with _MODE_LOCK:
        _check_mode_locked(agent, mode)
        agent.add_activation(callback)
        setattr(agent, _MODE_ATTRIBUTE, mode)
