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

"""Private synchronous boundaries, not production component implementations.

I/O belongs in blocking preparation, subscription workers or ordinary tool
activities. Async callers must offload it; deterministic workflow bodies must
not call these I/O methods. The router client owns loop-affine MCP adaptation.

All returned models are detached caller-owned snapshots. Implementations must
also detach retained inputs. Dependency injection borrows resources: the
composition owner closes the router client, not tools or the manager.
The agent retains ownership of its configured state-store primitive.
"""

from __future__ import annotations

from typing import Literal, Protocol

from drasi_agent_router_contracts import (
    ListQueriesResponse,
    SubscribeRequest,
    SubscribeResponse,
    UnsubscribeRequest,
    UnsubscribeResponse,
)
from drasi_agent_router_contracts.models.Operation import Operation

from ._models import (
    IntentDocument,
    IntentSnapshot,
    RouterFailureCategory,
    SubscriptionIntent,
    SubscriptionScope,
    router_failure_outcome,
)


class RouterError(Exception):
    """A classified router failure; messages contain no server payload."""

    def __init__(self, category: RouterFailureCategory) -> None:
        self.category = category
        super().__init__(f"Drasi router operation failed: {category}.")

    @property
    def mutation_outcome(self) -> Literal["rejected", "uncertain"]:
        return router_failure_outcome(self.category)


class IntentStoreError(Exception):
    """A failed read/write, never equivalent to an absent record.

    Unavailable writes may have committed. Reload before deciding what to do;
    a conflict must never trigger a blind retry of the stale whole document.
    """

    def __init__(
        self,
        category: Literal["unavailable", "corrupt", "unsupported_version", "conflict"],
    ) -> None:
        self.category = category
        super().__init__(f"Drasi intent store operation failed: {category}.")


class SubscriptionCommandError(Exception):
    """A local command rejection with a safe, actionable category."""

    def __init__(
        self,
        category: Literal[
            "invalid_input", "unknown_query", "pending_operation", "unavailable"
        ],
    ) -> None:
        self.category = category
        super().__init__(f"Drasi subscription command rejected: {category}.")


class RouterClient(Protocol):
    """Bound to one scope; the composition owner manages the client lifetime.

    The client validates shared wire models, MCP isError, identity and reply
    correlation (query/incarnation/operation SET/derived topic). Only validated
    confirmations return normally. All transport/protocol failures raise
    RouterError; a malformed mutation reply is uncertain, not a rejection.
    No instructions are sent to this interface.
    """

    @property
    def scope(self) -> SubscriptionScope: ...

    def list_queries(self) -> ListQueriesResponse:
        """Prepare/cache the complete startup catalog; empty is a valid result."""
        ...

    def subscribe(self, request: SubscribeRequest) -> SubscribeResponse:
        """Confirm created/updated state for the request's exact scoped identity."""
        ...

    def unsubscribe(self, request: UnsubscribeRequest) -> UnsubscribeResponse:
        """Both removed=True and removed=False confirm absence for this lifecycle."""
        ...

    def close(self) -> None:
        """Idempotently release runtime resources; never unsubscribe durable rules."""
        ...


class IntentReader(Protocol):
    """Scoped read-only view; get may perform blocking persistent I/O."""

    @property
    def scope(self) -> SubscriptionScope: ...

    def get(self, query_id: str) -> SubscriptionIntent | None:
        """Return an owned copy, or None ONLY for absence; otherwise raise IntentStoreError."""
        ...


class IntentRepository(IntentReader, Protocol):
    """One document, one ETag shared by every query in this scope.

    Uses the configured AgentStateConfig.store primitive. Wrong scope,
    corrupt data and unsupported versions must fail explicitly, not disappear.
    """

    def load(self) -> IntentSnapshot | None:
        """Load all intent, including retired queries; None means document absent."""
        ...

    def initialize(self, document: IntentDocument) -> None:
        """Unconditional initialization during exclusive single-owner preparation.

        The caller first establishes absence and excludes all other work.
        This is NOT atomic create-if-absent and is never a conflict fallback.
        Reload to obtain the first ETag before admitting work.
        """
        ...

    def save(self, document: IntentDocument, *, expected_etag: str) -> None:
        """Replace the WHOLE document conditionally; return only after confirmation.

        Any query's successful write invalidates all older document ETags.
        On conflict, the manager merges only its intended change into fresh
        state before retrying. Never retry this stale replacement unchanged.
        Reload after success to obtain the new ETag; save itself returns none.
        """
        ...


class SubscriptionManager(Protocol):
    """Local commands over borrowed repository/client dependencies.

    Commands raise SubscriptionCommandError, RouterError or IntentStoreError.
    Serialize unresolved same-query transitions; do not overwrite them.
    Router mutations follow durable pending writes; success follows the final
    local commit. Retries/updates retain incarnation, a new lifecycle does not.
    """

    def subscribe(
        self, query_id: str, *, operations: tuple[Operation, ...], instructions: str
    ) -> SubscriptionIntent:
        """Validate nonempty unique operations/instructions; return persisted active intent."""
        ...

    def unsubscribe(self, query_id: str) -> None:
        """Persist pending-unsubscribe before RPC; remove intent only after confirmation.

        Already absent local intent is a successful no-op. An unresolved failure
        retains pending intent. Retired catalog queries can still unsubscribe.
        """
        ...

    def list_subscriptions(self) -> tuple[SubscriptionIntent, ...]:
        """Return owned local intent snapshots, not an assertion of live router health."""
        ...

    def reconcile(self, catalog: ListQueriesResponse) -> None:
        """Prepare/cache the catalog and reconcile before admitting any normal work.

        Finish pending unsubscriptions even for retired queries; otherwise
        retain removed queries as unavailable, without automatic revival.
        Reassert active/pending subscriptions using their existing incarnation.
        Retain an owned catalog copy. No periodic refresh or startup model turn.
        """
        ...
