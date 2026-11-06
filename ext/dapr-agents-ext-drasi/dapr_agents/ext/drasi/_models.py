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

"""Private configuration and durable intent for Drasi subscriptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias
from urllib.parse import quote

from drasi_agent_router_contracts import (
    ListQueriesResponse,
    Subscriber,
    agent_dead_letter_topic,
    agent_inbox_topic,
    parse,
    to_wire,
)
from drasi_agent_router_contracts.models.Operation import Operation
from jsonschema.exceptions import ValidationError as WireValidationError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

NonEmptyString = Annotated[StrictStr, Field(min_length=1)]
SubscriptionStatus: TypeAlias = Literal[
    "pending_subscribe",
    "pending_update",
    "active",
    "pending_unsubscribe",
    "unavailable",
]
RouterFailureCategory: TypeAlias = Literal[
    "invalid_arguments",
    "unknown_query",
    "incarnation_conflict",
    "state_unavailable",
    "transport",
    "invalid_response",
]
MutationOutcome: TypeAlias = Literal["confirmed", "rejected", "uncertain"]


def router_failure_outcome(
    category: RouterFailureCategory,
) -> Literal["rejected", "uncertain"]:
    """Classify mutation confirmation, not whether retrying is appropriate."""
    if category in ("invalid_arguments", "unknown_query", "incarnation_conflict"):
        return "rejected"
    return "uncertain"


class SubscriptionScope(BaseModel):
    """Exact, immutable identity; wire subscribers are fresh shared models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    router_id: StrictStr
    namespace: StrictStr
    app_id: StrictStr
    agent_name: StrictStr

    @property
    def subscriber(self) -> Subscriber:
        return parse(
            Subscriber,
            {
                "namespace": self.namespace,
                "app_id": self.app_id,
                "agent_name": self.agent_name,
            },
        )

    @property
    def inbox_topic(self) -> str:
        return agent_inbox_topic(self.router_id, self.subscriber)

    @property
    def dead_letter_topic(self) -> str:
        return agent_dead_letter_topic(self.router_id, self.subscriber)

    @model_validator(mode="after")
    def validate_identity(self) -> SubscriptionScope:
        try:
            agent_inbox_topic(self.router_id, self.subscriber)
        except WireValidationError as error:
            raise ValueError("Invalid router/subscriber identity.") from error
        return self


class ResolvedDrasiConfig(BaseModel):
    """Resolved coordinates only; the composition owner supplies the resources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: SubscriptionScope
    pubsub_name: NonEmptyString
    state_store_name: NonEmptyString
    workflow_name: NonEmptyString
    dapr_http_port: Annotated[StrictInt, Field(ge=1, le=65535)]

    @field_validator("pubsub_name", "state_store_name", "workflow_name")
    @classmethod
    def reject_blank_names(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Resolved resource names must not be blank.")
        return value

    @property
    def router_mcp_url(self) -> str:
        namespace, app_id = self.scope.router_id.split("/")
        target = quote(f"{app_id}.{namespace}", safe="")
        return f"http://localhost:{self.dapr_http_port}/v1.0/invoke/{target}/method/mcp"


class RouterOperationOutcome(BaseModel):
    """Safe local outcome metadata, never raw transport errors or payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["subscribe", "unsubscribe"]
    outcome: MutationOutcome
    error_category: RouterFailureCategory | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> RouterOperationOutcome:
        expected = (
            "confirmed"
            if self.error_category is None
            else router_failure_outcome(self.error_category)
        )
        if self.outcome != expected:
            raise ValueError("Router outcome does not match its error category.")
        return self


class SubscriptionIntent(BaseModel):
    """Caller-owned intent snapshot, including the historical full catalog."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    query_id: NonEmptyString
    operations: Annotated[tuple[Operation, ...], Field(min_length=1, max_length=3)]
    instructions: NonEmptyString = Field(repr=False)
    catalog_snapshot: ListQueriesResponse = Field(repr=False)
    incarnation: NonEmptyString
    status: SubscriptionStatus
    last_outcome: RouterOperationOutcome | None = None

    @field_validator("operations")
    @classmethod
    def reject_duplicate_operations(
        cls, value: tuple[Operation, ...]
    ) -> tuple[Operation, ...]:
        if len(value) != len(set(value)):
            raise ValueError("Operations must be a non-empty unique subset of i/u/d.")
        return value

    @field_validator("instructions")
    @classmethod
    def reject_blank_instructions(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Handling instructions must not be blank.")
        return value

    @field_validator("catalog_snapshot", mode="before")
    @classmethod
    def parse_catalog_snapshot(cls, value: Any) -> ListQueriesResponse:
        try:
            document = (
                to_wire(value) if isinstance(value, ListQueriesResponse) else value
            )
            return parse(ListQueriesResponse, document)
        except (WireValidationError, ValueError) as error:
            raise ValueError("Invalid router catalog snapshot.") from error

    @field_serializer("catalog_snapshot")
    def serialize_catalog_snapshot(self, value: ListQueriesResponse) -> dict[str, Any]:
        return to_wire(value)

    @model_validator(mode="after")
    def validate_catalog_query(self) -> SubscriptionIntent:
        if not any(
            query.query_id == self.query_id for query in self.catalog_snapshot.queries
        ):
            raise ValueError(
                "Intent query is missing from its stored catalog snapshot."
            )
        return self


class IntentDocument(BaseModel):
    """One versioned document for a router/subscriber, not one key per query."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    format_version: Literal[1]
    scope: SubscriptionScope
    intents: dict[StrictStr, SubscriptionIntent] = Field(repr=False)

    @field_validator("format_version", mode="before")
    @classmethod
    def validate_format_version(cls, value: Any) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported intent document format version.")
        return value

    @field_validator("intents")
    @classmethod
    def detach_intents(
        cls, value: dict[str, SubscriptionIntent]
    ) -> dict[str, SubscriptionIntent]:
        return {
            query_id: intent.model_copy(deep=True) for query_id, intent in value.items()
        }

    @model_validator(mode="after")
    def validate_scope(self) -> IntentDocument:
        for query_id, intent in self.intents.items():
            if query_id != intent.query_id:
                raise ValueError("Intent key does not match its query ID.")
            if intent.catalog_snapshot.router_id != self.scope.router_id:
                raise ValueError("Intent catalog belongs to a different router.")
        return self


@dataclass(frozen=True)
class IntentSnapshot:
    """An owned document copy with the ETag of the ENTIRE scoped document."""

    document: IntentDocument = field(repr=False)
    etag: str
