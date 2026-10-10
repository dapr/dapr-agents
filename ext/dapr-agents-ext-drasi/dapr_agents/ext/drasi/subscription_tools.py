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

"""Private tool construction over a prepared catalog and borrowed manager.

The returned synchronous tools belong in ordinary tool activities. This module
does not attach tools, prepare resources, or change agent prompts. Its diagnostics
exclude payloads; core console output and telemetry are separate concerns.
"""

import hashlib
import json
import logging
import re

from drasi_agent_router_contracts import ListQueriesResponse, parse, to_wire
from drasi_agent_router_contracts.models.Operation import Operation
from drasi_agent_router_contracts.models.Query import Query
from jsonschema.exceptions import ValidationError as WireValidationError
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from dapr_agents.tool import AgentTool
from dapr_agents.tool.executor import AgentToolExecutor
from dapr_agents.types import ToolResult

from ._interfaces import (
    IntentStoreError,
    RouterError,
    SubscriptionCommandError,
    SubscriptionManager,
)

logger = logging.getLogger(__name__)

_LIST_TOOL_NAME = "list_drasi_subscriptions"
_MAX_TOOL_NAME_LENGTH = 64
_CHANGE_SEMANTICS = (
    "Operations describe projected query-result rows: i means entering the result "
    "set (after), u means a matching row changed (before and after), and d means "
    "leaving the result set (before). These are not necessarily database row "
    "inserts, updates, or deletes. Monitoring persists across tasks and ordinary "
    "restarts until explicitly unsubscribed. There is no historical catch-up or "
    "current-result snapshot; delayed or duplicate deliveries can still occur."
)


class _SubscribeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    operations: list[Operation] = Field(
        min_length=1,
        max_length=3,
        json_schema_extra={"uniqueItems": True},
        description=(
            "Explicit non-empty selection of distinct result-change operations: "
            "i (enter), u (change), d (leave). No default operation filter."
        ),
    )
    instructions: StrictStr = Field(
        min_length=1,
        pattern=r"\S",
        repr=False,
        description=(
            "Self-contained handling instructions for each independent event "
            "workflow: include the objective, actions, and constraints without "
            "relying on this conversation. Make external actions idempotent "
            "because duplicate event workflows are possible."
        ),
    )

    @field_validator("operations")
    @classmethod
    def reject_duplicate_operations(
        cls, operations: list[Operation]
    ) -> list[Operation]:
        if len(operations) != len(set(operations)):
            raise ValueError("Operations must be a unique subset of i/u/d.")
        return operations


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


def _tool_name(command: str, query_id: str) -> str:
    digest = hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-z0-9_-]", "", query_id.lower()) or "query"
    budget = _MAX_TOOL_NAME_LENGTH - len(command) - len(digest) - 2
    return f"{command}_{slug[:budget]}_{digest}"


def _query_description(query: Query) -> str:
    description = (
        f"Query: {query.title}\nQuery ID: {query.query_id}\n{query.description}"
    )
    if query.usage is not None:
        description += f"\nUsage: {query.usage}"
    return f"{description}\n{_CHANGE_SEMANTICS}"


def _command_failure(
    command: str,
    error: SubscriptionCommandError | RouterError | IntentStoreError,
) -> ToolResult:
    if isinstance(error, SubscriptionCommandError):
        source = "command"
        detail = {
            "invalid_input": (
                "Provide distinct operations from i/u/d and non-blank, "
                "self-contained handling instructions."
            ),
            "unknown_query": "The query is not available in the prepared catalog.",
            "pending_operation": (
                "A previous change is unresolved. Inspect local subscription "
                "status and allow recovery before issuing another change."
            ),
            "unavailable": (
                "The subscription is unavailable. Inspect local subscription "
                "status. If retained intent is marked unavailable, unsubscribe "
                "first, then subscribe again. Otherwise check preparation and "
                "the operator's configuration."
            ),
        }[error.category]
    elif isinstance(error, RouterError):
        source = "router"
        detail = f"Router operation failed ({error.category})."
        if command != "list" and error.mutation_outcome == "uncertain":
            detail += (
                " The change may have taken effect and local intent may remain "
                "pending. Inspect local status after recovery."
            )
    else:
        source = "intent_store"
        if command == "list":
            detail = (
                f"Local subscription state read failed ({error.category}). "
                "Retry listing after storage recovery."
            )
        else:
            detail = (
                f"Local subscription state operation failed ({error.category}). "
                "Inspect local status after storage recovery; do not assume that "
                "an attempted change was rolled back."
            )
    logger.error(
        "Drasi subscription %s failed (source=%s, category=%s).",
        command,
        source,
        error.category,
    )
    return ToolResult.error(f"Drasi {command} failed: {detail}")


def _query_tools(query: Query, manager: SubscriptionManager) -> list[AgentTool]:
    query_id = query.query_id
    description = _query_description(query)

    def subscribe(operations: list[Operation], instructions: str) -> ToolResult:
        try:
            intent = manager.subscribe(
                query_id, operations=tuple(operations), instructions=instructions
            )
        except (SubscriptionCommandError, RouterError, IntentStoreError) as error:
            return _command_failure("subscribe", error)
        selected = [operation.value for operation in intent.operations]
        return ToolResult.success(
            {
                "query_id": intent.query_id,
                "operations": selected,
                "status": intent.status,
            },
            text=(
                f"Subscription for {intent.query_id!r}: {intent.status}; "
                f"operations: {', '.join(selected)}. Handling instructions are "
                "saved for independent event workflows."
            ),
        )

    def unsubscribe() -> ToolResult:
        try:
            manager.unsubscribe(query_id)
        except (SubscriptionCommandError, RouterError, IntentStoreError) as error:
            return _command_failure("unsubscribe", error)
        return ToolResult.success(
            {"query_id": query_id, "subscribed": False},
            text=(
                f"Monitoring for {query_id!r} is stopped (already absent is also "
                "success). Already scheduled workflows are not cancelled."
            ),
        )

    return [
        AgentTool(
            name=_tool_name("subscribe", query_id),
            description=(
                "Start or update persistent monitoring for this query when the "
                "task calls for ongoing observation. Each admitted change starts "
                "an independent workflow using the supplied instructions. "
                "Repeating subscribe idempotently replaces the existing "
                "same-query operation filter and instructions; it does not add "
                f"another subscription.\n{description}"
            ),
            args_model=_SubscribeArgs,
            func=subscribe,
        ),
        AgentTool(
            name=_tool_name("unsubscribe", query_id),
            description=(
                "Stop persistent monitoring for this query. Already absent "
                "local intent is an idempotent success. This does not cancel "
                f"already scheduled workflows.\n{description}"
            ),
            args_model=_NoArgs,
            func=unsubscribe,
        ),
    ]


def build_subscription_tools(
    catalog: ListQueriesResponse, manager: SubscriptionManager
) -> list[AgentTool]:
    """Build tools without network/state I/O or agent mutation.

    The catalog is validated and detached using the shared protocol package.
    Query IDs are bound into closures, never exposed as model-selected arguments.
    The caller owns attachment and lifetime, including checking collisions against the
    agent's existing executor before attaching any of these tools.
    """
    try:
        prepared_catalog = parse(ListQueriesResponse, to_wire(catalog))
    except (WireValidationError, ValueError):
        logger.error("Invalid Drasi subscription tool catalog.")
        raise ValueError("Invalid Drasi subscription tool catalog.") from None

    def list_subscriptions() -> ToolResult:
        try:
            intents = manager.list_subscriptions()
        except (SubscriptionCommandError, RouterError, IntentStoreError) as error:
            return _command_failure("list", error)
        subscriptions = [
            {
                "query_id": intent.query_id,
                "operations": [operation.value for operation in intent.operations],
                "instructions": intent.instructions,
                "status": intent.status,
            }
            for intent in intents
        ]
        result = {"source": "local_intent", "subscriptions": subscriptions}
        return ToolResult.success(
            result,
            text=(
                "Local subscription intent only, not live router state:\n"
                + json.dumps(result, ensure_ascii=True, indent=2)
            ),
        )

    tools = [
        tool
        for query in prepared_catalog.queries
        for tool in _query_tools(query, manager)
    ]
    tools.append(
        AgentTool(
            name=_LIST_TOOL_NAME,
            description=(
                "Inspect locally persisted Drasi subscription intent: query IDs, "
                "operation filters, handling instructions, and active, pending, "
                "or unavailable status. This does not query or verify live router "
                "state. An empty list means no local intent, not a failed read."
            ),
            args_model=_NoArgs,
            func=list_subscriptions,
        )
    )
    names: set[str] = set()
    for tool in tools:
        key = AgentToolExecutor._normalize(tool.name)
        if key in names:
            logger.error("Drasi generated tool name collision: %s.", tool.name)
            raise ValueError(f"Drasi generated tool name collision: {tool.name}.")
        names.add(key)
    return tools
