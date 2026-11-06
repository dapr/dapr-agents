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

"""Small deterministic fakes shared by Drasi subscription tests."""

# Shared fakes must satisfy their protocols even when tests.* ignores type errors.
# mypy: ignore-errors=False

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal, TypeAlias

from drasi_agent_router_contracts import (
    ListQueriesResponse,
    SubscribeRequest,
    SubscribeResponse,
    UnsubscribeRequest,
    UnsubscribeResponse,
    parse,
    to_wire,
)
from drasi_agent_router_contracts.models.Operation import Operation

from dapr_agents.ext.drasi._interfaces import (
    IntentRepository,
    IntentStoreError,
    RouterClient,
    RouterError,
    SubscriptionCommandError,
    SubscriptionManager,
)
from dapr_agents.ext.drasi._models import (
    IntentDocument,
    IntentSnapshot,
    RouterFailureCategory,
    SubscriptionIntent,
    SubscriptionScope,
)

ManagerError: TypeAlias = SubscriptionCommandError | RouterError | IntentStoreError
StoreOperation: TypeAlias = Literal["load", "get", "initialize", "save"]
StoreErrorCategory: TypeAlias = Literal[
    "unavailable", "corrupt", "unsupported_version", "conflict"
]


def _copy_catalog(catalog: ListQueriesResponse) -> ListQueriesResponse:
    return parse(ListQueriesResponse, to_wire(catalog))


def _copy_subscribe_request(request: SubscribeRequest) -> SubscribeRequest:
    return parse(SubscribeRequest, to_wire(request))


def _copy_subscribe_response(response: SubscribeResponse) -> SubscribeResponse:
    return parse(SubscribeResponse, to_wire(response))


def _copy_unsubscribe_request(request: UnsubscribeRequest) -> UnsubscribeRequest:
    return parse(UnsubscribeRequest, to_wire(request))


def _copy_unsubscribe_response(response: UnsubscribeResponse) -> UnsubscribeResponse:
    return parse(UnsubscribeResponse, to_wire(response))


def _copy_intent(intent: SubscriptionIntent) -> SubscriptionIntent:
    return SubscriptionIntent.model_validate(intent.model_dump(mode="json"))


def _copy_document(document: IntentDocument) -> IntentDocument:
    return IntentDocument.model_validate(document.model_dump(mode="json"))


@dataclass(frozen=True)
class SubscribeCall:
    query_id: str
    operations: tuple[Operation, ...]
    instructions: str = field(repr=False)


class ScriptedRouterClient(RouterClient):
    """A bound router fake whose replies must be explicitly scripted."""

    def __init__(self, scope: SubscriptionScope) -> None:
        self._scope = scope
        self._list_results: deque[ListQueriesResponse | RouterError] = deque()
        self._subscribe_results: deque[SubscribeResponse | RouterError] = deque()
        self._unsubscribe_results: deque[UnsubscribeResponse | RouterError] = deque()
        self._subscribe_requests: list[SubscribeRequest] = []
        self._unsubscribe_requests: list[UnsubscribeRequest] = []
        self.closed = False

    @property
    def scope(self) -> SubscriptionScope:
        return self._scope

    @property
    def subscribe_requests(self) -> tuple[SubscribeRequest, ...]:
        return tuple(_copy_subscribe_request(item) for item in self._subscribe_requests)

    @property
    def unsubscribe_requests(self) -> tuple[UnsubscribeRequest, ...]:
        return tuple(
            _copy_unsubscribe_request(item) for item in self._unsubscribe_requests
        )

    def queue_catalog(self, catalog: ListQueriesResponse) -> None:
        self._list_results.append(_copy_catalog(catalog))

    def queue_list_error(self, category: RouterFailureCategory) -> None:
        self._list_results.append(RouterError(category))

    def queue_subscribe(self, response: SubscribeResponse) -> None:
        self._subscribe_results.append(_copy_subscribe_response(response))

    def queue_subscribe_error(self, category: RouterFailureCategory) -> None:
        self._subscribe_results.append(RouterError(category))

    def queue_unsubscribe(self, response: UnsubscribeResponse) -> None:
        self._unsubscribe_results.append(_copy_unsubscribe_response(response))

    def queue_unsubscribe_error(self, category: RouterFailureCategory) -> None:
        self._unsubscribe_results.append(RouterError(category))

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Scripted router client is closed.")

    def list_queries(self) -> ListQueriesResponse:
        self._require_open()
        if not self._list_results:
            raise AssertionError("No scripted list-queries result.")
        result = self._list_results.popleft()
        if isinstance(result, RouterError):
            raise result
        return _copy_catalog(result)

    def subscribe(self, request: SubscribeRequest) -> SubscribeResponse:
        self._require_open()
        self._subscribe_requests.append(_copy_subscribe_request(request))
        if not self._subscribe_results:
            raise AssertionError("No scripted subscribe result.")
        result = self._subscribe_results.popleft()
        if isinstance(result, RouterError):
            raise result
        return _copy_subscribe_response(result)

    def unsubscribe(self, request: UnsubscribeRequest) -> UnsubscribeResponse:
        self._require_open()
        self._unsubscribe_requests.append(_copy_unsubscribe_request(request))
        if not self._unsubscribe_results:
            raise AssertionError("No scripted unsubscribe result.")
        result = self._unsubscribe_results.popleft()
        if isinstance(result, RouterError):
            raise result
        return _copy_unsubscribe_response(result)

    def close(self) -> None:
        self.closed = True


class ScriptedSubscriptionManager(SubscriptionManager):
    """A command-boundary fake without subscription lifecycle behavior."""

    def __init__(self) -> None:
        self._subscribe_results: deque[SubscriptionIntent | ManagerError] = deque()
        self._unsubscribe_results: deque[ManagerError | None] = deque()
        self._reconcile_results: deque[ManagerError | None] = deque()
        self._list_errors: deque[ManagerError] = deque()
        self._subscriptions: tuple[SubscriptionIntent, ...] | None = None
        self._subscribe_calls: list[SubscribeCall] = []
        self._unsubscribe_calls: list[str] = []
        self._reconciled_catalogs: list[ListQueriesResponse] = []

    @property
    def subscribe_calls(self) -> tuple[SubscribeCall, ...]:
        return tuple(self._subscribe_calls)

    @property
    def unsubscribe_calls(self) -> tuple[str, ...]:
        return tuple(self._unsubscribe_calls)

    @property
    def reconciled_catalogs(self) -> tuple[ListQueriesResponse, ...]:
        return tuple(_copy_catalog(item) for item in self._reconciled_catalogs)

    def queue_subscribe(self, intent: SubscriptionIntent) -> None:
        self._subscribe_results.append(_copy_intent(intent))

    def queue_subscribe_error(self, error: ManagerError) -> None:
        self._subscribe_results.append(error)

    def queue_unsubscribe(self, error: ManagerError | None = None) -> None:
        self._unsubscribe_results.append(error)

    def queue_reconcile(self, error: ManagerError | None = None) -> None:
        self._reconcile_results.append(error)

    def queue_list_error(self, error: ManagerError) -> None:
        self._list_errors.append(error)

    def set_subscriptions(self, subscriptions: tuple[SubscriptionIntent, ...]) -> None:
        self._subscriptions = tuple(_copy_intent(item) for item in subscriptions)

    def subscribe(
        self, query_id: str, *, operations: tuple[Operation, ...], instructions: str
    ) -> SubscriptionIntent:
        self._subscribe_calls.append(
            SubscribeCall(
                query_id=query_id,
                operations=tuple(operations),
                instructions=instructions,
            )
        )
        if not self._subscribe_results:
            raise AssertionError("No scripted manager subscribe result.")
        result = self._subscribe_results.popleft()
        if isinstance(
            result, (SubscriptionCommandError, RouterError, IntentStoreError)
        ):
            raise result
        return _copy_intent(result)

    def unsubscribe(self, query_id: str) -> None:
        self._unsubscribe_calls.append(query_id)
        if not self._unsubscribe_results:
            raise AssertionError("No scripted manager unsubscribe result.")
        error = self._unsubscribe_results.popleft()
        if error is not None:
            raise error

    def list_subscriptions(self) -> tuple[SubscriptionIntent, ...]:
        if self._list_errors:
            raise self._list_errors.popleft()
        if self._subscriptions is None:
            raise AssertionError("No scripted manager subscription listing.")
        return tuple(_copy_intent(item) for item in self._subscriptions)

    def reconcile(self, catalog: ListQueriesResponse) -> None:
        self._reconciled_catalogs.append(_copy_catalog(catalog))
        if not self._reconcile_results:
            raise AssertionError("No scripted manager reconcile result.")
        error = self._reconcile_results.popleft()
        if error is not None:
            raise error


class InMemoryIntentRepository(IntentRepository):
    """One scoped document with a single whole-document ETag."""

    def __init__(self, scope: SubscriptionScope) -> None:
        self._scope = scope
        self._document: IntentDocument | None = None
        self._revision = 0
        self._lock = Lock()
        self._failures: dict[StoreOperation, deque[IntentStoreError]] = defaultdict(
            deque
        )

    @property
    def scope(self) -> SubscriptionScope:
        return self._scope

    def fail_next(
        self, operation: StoreOperation, category: StoreErrorCategory
    ) -> None:
        self._failures[operation].append(IntentStoreError(category))

    def _raise_scripted_failure(self, operation: StoreOperation) -> None:
        failures = self._failures[operation]
        if failures:
            raise failures.popleft()

    def _etag(self) -> str:
        return f"intent-document-{self._revision}"

    def _validate_scope(self, document: IntentDocument) -> None:
        if document.scope != self._scope:
            raise ValueError("Intent document scope does not match repository scope.")

    def load(self) -> IntentSnapshot | None:
        with self._lock:
            self._raise_scripted_failure("load")
            if self._document is None:
                return None
            return IntentSnapshot(
                document=_copy_document(self._document),
                etag=self._etag(),
            )

    def get(self, query_id: str) -> SubscriptionIntent | None:
        with self._lock:
            self._raise_scripted_failure("get")
            if self._document is None:
                return None
            intent = self._document.intents.get(query_id)
            return None if intent is None else _copy_intent(intent)

    def initialize(self, document: IntentDocument) -> None:
        with self._lock:
            self._raise_scripted_failure("initialize")
            self._validate_scope(document)
            self._document = _copy_document(document)
            self._revision += 1

    def save(self, document: IntentDocument, *, expected_etag: str) -> None:
        if not expected_etag:
            raise ValueError("expected_etag must be nonempty.")
        with self._lock:
            self._raise_scripted_failure("save")
            self._validate_scope(document)
            if self._document is None or expected_etag != self._etag():
                raise IntentStoreError("conflict")
            self._document = _copy_document(document)
            self._revision += 1
