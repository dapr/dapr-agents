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

"""
Lifecycle dispatch extension point for DurableAgent.

This module exposes a minimal Protocol that DurableAgent calls into at
known lifecycle events (BEFORE_AGENT_INVOKE, BEFORE_TOOL_CALL, etc.).
Implementations of this Protocol can intercept those events to add
authentication, OBO, HITL approver authorization, observability, and
other cross-cutting concerns without modifying DurableAgent itself.

Design notes:
    - dapr_agents.lifecycle stays OSS-clean: no proprietary or
      plugin-specific imports. Any implementation can plug in.
    - DecisionDict mirrors the serializable fields of dapr_agents.hooks
      decisions, plus a few forward-looking approver-authorization fields
      that dispatcher implementations may populate (see below).
      Implementations that want richer return types can still subclass;
      this type is the minimum protocol contract.
    - This module is intentionally tiny. Heavy plugin logic lives
      outside dapr-agents.
"""

from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

# Required, and the TypedDict that honors it, live in typing_extensions until
# Python 3.11; on the project's 3.10 floor the stdlib typing.TypedDict ignores
# the Required marker at runtime, so both must come from typing_extensions.
from typing_extensions import Required, TypedDict


class DecisionDict(TypedDict, total=False):
    """
    Serializable decision returned from LifecycleDispatcher.dispatch.

    Fields up to and including `details` mirror the serializable fields of
    the dapr_agents.hooks decisions. The `type` field is required. Other
    fields are populated conditionally based on the decision type:
        - "proceed": no other fields
        - "skip": optional `result`
        - "mutate": `payload` dict
        - "require_approval": optional `timeout_seconds`, `instructions`,
          `reason`
        - "deny": optional `reason`, `code`, `details`

    The `required_approver_scopes`, `allowed_approver_subjects`, and
    `approver_audience` fields are extension fields for dispatcher
    implementations that perform approver authorization. They are not part
    of the current dapr_agents.hooks decisions or approval event schema, so
    the core agent runtime does not interpret them; a dispatcher that emits
    them is responsible for acting on them.
    """

    type: Required[Literal["proceed", "skip", "mutate", "require_approval", "deny"]]
    result: Any
    payload: Dict[str, Any]
    reason: Optional[str]
    code: Optional[str]
    details: Optional[Dict[str, Any]]
    timeout_seconds: Optional[int]
    instructions: Optional[str]
    required_approver_scopes: list[str]
    allowed_approver_subjects: Optional[list[str]]
    approver_audience: Optional[str]


@runtime_checkable
class LifecycleDispatcher(Protocol):
    """
    Protocol implemented by anything that wants to intercept DurableAgent
    lifecycle events.

    DurableAgent calls dispatch() at known lifecycle points. Returning None
    or {"type": "proceed"} means "run the step normally." Returning a
    DecisionDict with type "skip" | "mutate" | "require_approval" | "deny"
    alters the workflow per the existing hook semantics in
    dapr_agents.hooks.

    attach() is called once during DurableAgent construction so the
    dispatcher can initialize any per-agent state (open clients, register
    sub-handlers, etc.). detach() is called during cleanup. Implementations
    are free to no-op either method.
    """

    def attach(self, agent: Any) -> None:
        """
        Called by DurableAgent during initialization. The agent reference is
        passed in case the dispatcher needs to read agent config or register
        callbacks. Implementations should not retain the reference longer
        than necessary; detach() should release any held state.
        """
        ...

    def detach(self) -> None:
        """
        Called by DurableAgent during cleanup. Implementations release any
        per-agent resources here.
        """
        ...

    def dispatch(
        self,
        event_name: str,
        context: Dict[str, Any],
    ) -> Optional[DecisionDict]:
        """
        Called by DurableAgent at a known lifecycle event.

        Args:
            event_name: One of "BEFORE_AGENT_INVOKE", "BEFORE_TOOL_CALL",
                "BEFORE_LLM_CALL", "BEFORE_APPROVAL_DECISION",
                "AFTER_TOOL_CALL", "AFTER_LLM_CALL", "AFTER_AGENT_INVOKE",
                or future events added by extension.
            context: Event-specific context dict. Shape depends on
                event_name; documented in DurableAgent's dispatch sites.

        Returns:
            A DecisionDict to alter workflow behavior, or None for "proceed."
            Returning {"type": "proceed"} is equivalent to None.
        """
        ...
