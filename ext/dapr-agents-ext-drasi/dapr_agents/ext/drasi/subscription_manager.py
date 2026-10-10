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

"""Private subscription commands for preparation and ordinary tool activities."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Literal, NoReturn
from uuid import uuid4

from drasi_agent_router_contracts import (
    ListQueriesResponse,
    SubscribeRequest,
    UnsubscribeRequest,
    parse,
    parse_catalog,
    to_wire,
)
from drasi_agent_router_contracts.models.Operation import Operation
from jsonschema.exceptions import ValidationError as WireValidationError
from pydantic import ValidationError

from ._interfaces import (
    IntentRepository,
    IntentStoreError,
    RouterClient,
    RouterError,
    SubscriptionCommandError,
    SubscriptionManager,
)
from ._models import IntentSnapshot, RouterOperationOutcome, SubscriptionIntent

logger = logging.getLogger(__name__)

_MAX_CONFLICT_ATTEMPTS = 10
_PENDING_SUBSCRIPTIONS = {"pending_subscribe", "pending_update"}


class _UnavailableIntentError(SubscriptionCommandError):
    def __init__(self) -> None:
        super().__init__("unavailable")

    def __str__(self) -> str:
        return f"{super().__str__()} Unsubscribe first, then subscribe again."


class DrasiSubscriptionManager(SubscriptionManager):
    """Manage one scope using borrowed repository and router dependencies.

    The owner initializes the repository and calls ``reconcile`` before work.
    Commands are serialized within this manager; this is not a distributed lock.
    Listing remains available while a command waits for the router.
    """

    def __init__(self, *, repository: IntentRepository, router: RouterClient) -> None:
        if repository.scope != router.scope:
            logger.error("Drasi repository and router scopes must match.")
            raise ValueError("Drasi repository and router scopes must match.")
        self._repository = repository
        self._router = router
        self._scope = repository.scope
        self._catalog: ListQueriesResponse | None = None
        self._lock = Lock()

    def subscribe(
        self, query_id: str, *, operations: tuple[Operation, ...], instructions: str
    ) -> SubscriptionIntent:
        self._validate_query_id(query_id)
        with self._lock:
            current = self._load().document.intents.get(query_id)
            if current is not None and current.status == "unavailable":
                logger.error(
                    "Drasi intent is unavailable; unsubscribe first, then subscribe again."
                )
                raise _UnavailableIntentError() from None
            catalog = self._catalog
            if catalog is None:
                self._reject("unavailable")
            if not any(query.query_id == query_id for query in catalog.queries):
                self._reject("unknown_query")

            try:
                pending = SubscriptionIntent(
                    query_id=query_id,
                    operations=operations,
                    instructions=instructions,
                    catalog_snapshot=catalog,
                    incarnation=current.incarnation
                    if current is not None
                    else uuid4().hex,
                    status="pending_update"
                    if current is not None
                    else "pending_subscribe",
                )
            except ValidationError:
                self._reject("invalid_input")

            if current is not None:
                if current.status == "pending_unsubscribe":
                    self._reject("pending_operation")
                same_command = (
                    set(current.operations) == set(pending.operations)
                    and current.instructions == pending.instructions
                )
                if current.status in _PENDING_SUBSCRIPTIONS:
                    if not same_command:
                        self._reject("pending_operation")
                    pending = current.model_copy(deep=True)
                elif same_command:
                    return current

            self._commit(query_id, before=current, after=pending)
            return self._finish_subscribe(pending)

    def unsubscribe(self, query_id: str) -> None:
        self._validate_query_id(query_id)
        with self._lock:
            current = self._load().document.intents.get(query_id)
            if current is None:
                return
            if current.status in _PENDING_SUBSCRIPTIONS:
                self._reject("pending_operation")
            pending = (
                current
                if current.status == "pending_unsubscribe"
                else current.model_copy(
                    update={"status": "pending_unsubscribe", "last_outcome": None},
                    deep=True,
                )
            )
            self._commit(query_id, before=current, after=pending)
            self._finish_unsubscribe(pending)

    def list_subscriptions(self) -> tuple[SubscriptionIntent, ...]:
        return tuple(self._load().document.intents.values())

    def reconcile(self, catalog: ListQueriesResponse) -> None:
        with self._lock:
            self._catalog = None
            try:
                owned_catalog = parse_catalog(
                    catalog.model_dump(
                        mode="json", exclude_unset=True, warnings="error"
                    ),
                    self._scope.router_id,
                )
            except (WireValidationError, ValueError):
                logger.error("Drasi reconciliation received an invalid router catalog.")
                raise RouterError("invalid_response") from None

            query_ids = {query.query_id for query in owned_catalog.queries}
            snapshot = self._load()
            # Finish removals before reassertions that could fail and block startup.
            intents = sorted(
                snapshot.document.intents.values(),
                key=lambda intent: intent.status != "pending_unsubscribe",
            )
            for current in intents:
                if current.status == "pending_unsubscribe":
                    self._commit(current.query_id, before=current, after=current)
                    self._finish_unsubscribe(current)
                elif current.status == "unavailable":
                    continue
                elif current.query_id not in query_ids:
                    unavailable = current.model_copy(
                        update={"status": "unavailable"}, deep=True
                    )
                    self._commit(current.query_id, before=current, after=unavailable)
                else:
                    pending = (
                        current
                        if current.status in _PENDING_SUBSCRIPTIONS
                        else current.model_copy(
                            update={"status": "pending_update", "last_outcome": None},
                            deep=True,
                        )
                    )
                    self._commit(current.query_id, before=current, after=pending)
                    self._finish_subscribe(pending)
            self._catalog = owned_catalog

    def _finish_subscribe(self, pending: SubscriptionIntent) -> SubscriptionIntent:
        request = parse(
            SubscribeRequest,
            {
                "query_id": pending.query_id,
                "operations": [operation.value for operation in pending.operations],
                "subscriber": to_wire(self._scope.subscriber),
                "subscription_incarnation": pending.incarnation,
            },
        )
        try:
            self._router.subscribe(request)
        except RouterError as error:
            self._record_router_failure(pending, error, operation="subscribe")
            raise

        active = pending.model_copy(
            update={
                "status": "active",
                "last_outcome": RouterOperationOutcome(
                    operation="subscribe", outcome="confirmed"
                ),
            },
            deep=True,
        )
        self._commit(pending.query_id, before=pending, after=active)
        return active

    def _finish_unsubscribe(self, pending: SubscriptionIntent) -> None:
        request = parse(
            UnsubscribeRequest,
            {
                "query_id": pending.query_id,
                "subscriber": to_wire(self._scope.subscriber),
                "subscription_incarnation": pending.incarnation,
            },
        )
        try:
            self._router.unsubscribe(request)
        except RouterError as error:
            self._record_router_failure(pending, error, operation="unsubscribe")
            raise
        self._commit(pending.query_id, before=pending, after=None)

    def _record_router_failure(
        self,
        pending: SubscriptionIntent,
        error: RouterError,
        *,
        operation: Literal["subscribe", "unsubscribe"],
    ) -> None:
        logger.error(
            "Drasi router %s failed (%s), scope=%s.",
            operation,
            error.category,
            self._scope.inbox_topic,
        )
        failed = pending.model_copy(
            update={
                "last_outcome": RouterOperationOutcome(
                    operation=operation,
                    outcome=error.mutation_outcome,
                    error_category=error.category,
                )
            },
            deep=True,
        )
        self._commit(pending.query_id, before=pending, after=failed)

    def _load(self) -> IntentSnapshot:
        try:
            snapshot = self._repository.load()
        except IntentStoreError as error:
            logger.error("Drasi intent read failed (%s).", error.category)
            raise
        if snapshot is None or not snapshot.etag:
            self._store_error("unavailable")
        if snapshot.document.scope != self._scope:
            self._store_error("corrupt")
        return snapshot

    def _commit(
        self,
        query_id: str,
        *,
        before: SubscriptionIntent | None,
        after: SubscriptionIntent | None,
    ) -> None:
        for attempt in range(_MAX_CONFLICT_ATTEMPTS):
            snapshot = self._load()
            current = snapshot.document.intents.get(query_id)
            if current == after:
                return
            if current != before:
                self._store_error("conflict")
            if after is None:
                del snapshot.document.intents[query_id]
            else:
                snapshot.document.intents[query_id] = after.model_copy(deep=True)
            try:
                self._repository.save(snapshot.document, expected_etag=snapshot.etag)
                return
            except IntentStoreError as error:
                if (
                    error.category != "conflict"
                    or attempt == _MAX_CONFLICT_ATTEMPTS - 1
                ):
                    logger.error("Drasi intent write failed (%s).", error.category)
                    raise
                logger.debug("Drasi intent write conflicted; reloading before merging.")

    def _validate_query_id(self, query_id: str) -> None:
        if not isinstance(query_id, str) or not query_id:
            self._reject("invalid_input")

    def _reject(
        self,
        category: Literal[
            "invalid_input", "unknown_query", "pending_operation", "unavailable"
        ],
    ) -> NoReturn:
        logger.error("Drasi subscription command rejected (%s).", category)
        raise SubscriptionCommandError(category) from None

    def _store_error(
        self, category: Literal["unavailable", "corrupt", "conflict"]
    ) -> NoReturn:
        logger.error("Drasi subscription intent failed (%s).", category)
        raise IntentStoreError(category)
