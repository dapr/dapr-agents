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

"""Subscription transitions and recovery over repository/router boundaries."""

# mypy: ignore-errors=False

from __future__ import annotations

import itertools
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Callable
from uuid import UUID

import pytest
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

from dapr_agents.ext.drasi import subscription_manager as manager_module
from dapr_agents.ext.drasi._interfaces import (
    IntentStoreError,
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
    SubscriptionStatus,
)
from dapr_agents.ext.drasi.subscription_manager import DrasiSubscriptionManager

from .fakes import InMemoryIntentRepository, ScriptedRouterClient, StoreErrorCategory

_QUERY = "service-errors"
_OPERATIONS = (Operation.i, Operation.u)
_INSTRUCTIONS = "PRIVATE-HANDLING-INSTRUCTIONS"


class _Repository(InMemoryIntentRepository):
    def __init__(self, scope: SubscriptionScope) -> None:
        super().__init__(scope)
        self.attempts = 0
        self.saved: list[IntentDocument] = []
        self.before_save: Callable[[IntentDocument], None] | None = None
        self.after_save: Callable[[IntentDocument], None] | None = None

    def save(self, document: IntentDocument, *, expected_etag: str) -> None:
        self.attempts += 1
        if self.before_save is not None:
            self.before_save(document)
        super().save(document, expected_etag=expected_etag)
        self.saved.append(document.model_copy(deep=True))
        if self.after_save is not None:
            self.after_save(document)


class _Router(ScriptedRouterClient):
    def __init__(self, scope: SubscriptionScope) -> None:
        super().__init__(scope)
        self.before_subscribe: Callable[[SubscribeRequest], None] | None = None
        self.before_unsubscribe: Callable[[UnsubscribeRequest], None] | None = None

    def subscribe(self, request: SubscribeRequest) -> SubscribeResponse:
        if self.before_subscribe is not None:
            self.before_subscribe(request)
        response = super().subscribe(request)
        assert response.query_id == request.query_id
        assert response.subscription_incarnation == request.subscription_incarnation
        assert set(response.operations) == set(request.operations)
        assert response.topic_name == self.scope.inbox_topic
        return response

    def unsubscribe(self, request: UnsubscribeRequest) -> UnsubscribeResponse:
        if self.before_unsubscribe is not None:
            self.before_unsubscribe(request)
        response = super().unsubscribe(request)
        assert response.query_id == request.query_id
        return response


@pytest.fixture(autouse=True)
def predictable_incarnations(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = itertools.count(1)

    def next_uuid() -> UUID:
        return UUID(int=next(counter))

    monkeypatch.setattr(manager_module, "uuid4", next_uuid)


def _incarnation(number: int = 1) -> str:
    return UUID(int=number).hex


@pytest.fixture
def repository(scope: SubscriptionScope) -> _Repository:
    repository = _Repository(scope)
    repository.initialize(IntentDocument(format_version=1, scope=scope, intents={}))
    return repository


@pytest.fixture
def router(scope: SubscriptionScope) -> _Router:
    return _Router(scope)


@pytest.fixture
def manager(
    repository: _Repository, router: _Router, catalog: ListQueriesResponse
) -> DrasiSubscriptionManager:
    manager = DrasiSubscriptionManager(repository=repository, router=router)
    manager.reconcile(catalog)
    return manager


def _confirm_subscribe(
    router: _Router,
    *,
    query_id: str = _QUERY,
    incarnation: str | None = None,
    operations: tuple[Operation, ...] = _OPERATIONS,
) -> None:
    router.queue_subscribe(
        parse(
            SubscribeResponse,
            {
                "query_id": query_id,
                "operations": [operation.value for operation in operations],
                "subscription_incarnation": incarnation or _incarnation(),
                "topic_name": router.scope.inbox_topic,
                "status": "updated",
            },
        )
    )


def _confirm_unsubscribe(
    router: _Router, *, query_id: str = _QUERY, removed: bool = True
) -> None:
    router.queue_unsubscribe(
        parse(UnsubscribeResponse, {"query_id": query_id, "removed": removed})
    )


def _subscribe(manager: DrasiSubscriptionManager) -> SubscriptionIntent:
    return manager.subscribe(_QUERY, operations=_OPERATIONS, instructions=_INSTRUCTIONS)


def _seed(repository: _Repository, *intents: SubscriptionIntent) -> None:
    repository.initialize(
        IntentDocument(
            format_version=1,
            scope=repository.scope,
            intents={intent.query_id: intent for intent in intents},
        )
    )


def test_subscribe_commits_pending_before_rpc_and_active_before_success(
    manager: DrasiSubscriptionManager, repository: _Repository, router: _Router
) -> None:
    def inspect_pending(request: SubscribeRequest) -> None:
        pending = repository.get(request.query_id)
        assert pending is not None
        assert pending.status == "pending_subscribe"
        assert pending.incarnation == request.subscription_incarnation
        assert pending.instructions == _INSTRUCTIONS

    router.before_subscribe = inspect_pending
    _confirm_subscribe(router)
    boundary: SubscriptionManager = manager
    active = boundary.subscribe(
        _QUERY, operations=_OPERATIONS, instructions=_INSTRUCTIONS
    )
    assert [document.intents[_QUERY].status for document in repository.saved] == [
        "pending_subscribe",
        "active",
    ]
    assert active == repository.get(_QUERY)
    assert active.last_outcome is not None
    assert active.last_outcome.outcome == "confirmed"
    request = to_wire(router.subscribe_requests[0])
    assert request["subscriber"] == to_wire(repository.scope.subscriber)
    assert _INSTRUCTIONS not in json.dumps(request)
    assert set(request) == {
        "query_id",
        "operations",
        "subscriber",
        "subscription_incarnation",
    }


def test_repeated_active_command_is_a_local_noop_and_updates_retain_incarnation(
    manager: DrasiSubscriptionManager, repository: _Repository, router: _Router
) -> None:
    _confirm_subscribe(router)
    first = _subscribe(manager)
    repeated = manager.subscribe(
        _QUERY, operations=tuple(reversed(_OPERATIONS)), instructions=_INSTRUCTIONS
    )
    assert repeated == first
    assert len(router.subscribe_requests) == 1
    assert repository.attempts == 2

    def inspect_update(request: SubscribeRequest) -> None:
        pending = repository.get(_QUERY)
        assert pending is not None
        assert pending.status == "pending_update"
        assert pending.incarnation == first.incarnation
        assert pending.instructions == "Updated instructions"

    router.before_subscribe = inspect_update
    _confirm_subscribe(router, operations=(Operation.d,))
    updated = manager.subscribe(
        _QUERY, operations=(Operation.d,), instructions="Updated instructions"
    )
    assert updated.status == "active"
    assert updated.incarnation == first.incarnation
    assert updated.operations == (Operation.d,)
    assert updated == repository.get(_QUERY)


@pytest.mark.parametrize("removed", (True, False))
def test_unsubscribe_removes_local_intent_only_after_confirmed_router_absence(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    removed: bool,
) -> None:
    _confirm_subscribe(router)
    active = _subscribe(manager)

    def inspect_pending(request: UnsubscribeRequest) -> None:
        pending = repository.get(_QUERY)
        assert pending is not None
        assert pending.status == "pending_unsubscribe"
        assert (
            pending.incarnation
            == active.incarnation
            == request.subscription_incarnation
        )

    router.before_unsubscribe = inspect_pending
    _confirm_unsubscribe(router, removed=removed)
    manager.unsubscribe(_QUERY)
    assert repository.get(_QUERY) is None
    snapshot = repository.load()
    assert snapshot is not None and snapshot.document.intents == {}
    assert repository.saved[-2].intents[_QUERY].status == "pending_unsubscribe"
    assert repository.saved[-1].intents == {}
    manager.unsubscribe(_QUERY)
    assert len(router.unsubscribe_requests) == 1


@pytest.mark.parametrize("operation", ("subscribe", "unsubscribe"))
@pytest.mark.parametrize(
    "category",
    (
        "transport",
        "invalid_response",
        "state_unavailable",
        "invalid_arguments",
        "unknown_query",
        "incarnation_conflict",
    ),
)
def test_router_failures_preserve_pending_intent_and_last_outcome(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    operation: str,
    category: RouterFailureCategory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if operation == "subscribe":
        router.queue_subscribe_error(category)
        pending_status = "pending_subscribe"
    else:
        _confirm_subscribe(router)
        _subscribe(manager)
        router.queue_unsubscribe_error(category)
        pending_status = "pending_unsubscribe"

    with pytest.raises(RouterError) as failure:
        if operation == "subscribe":
            _subscribe(manager)
        else:
            manager.unsubscribe(_QUERY)
    assert failure.value.category == category
    pending = repository.get(_QUERY)
    assert pending is not None
    assert pending.status == pending_status
    assert pending.incarnation == _incarnation()
    assert pending.last_outcome is not None
    assert pending.last_outcome.operation == operation
    assert pending.last_outcome.error_category == category
    assert pending.last_outcome.outcome == failure.value.mutation_outcome
    assert _INSTRUCTIONS not in caplog.text


def test_failed_update_retains_desired_pending_state_and_resumes_same_incarnation(
    manager: DrasiSubscriptionManager, repository: _Repository, router: _Router
) -> None:
    _confirm_subscribe(router)
    original = _subscribe(manager)
    router.queue_subscribe_error("transport")
    with pytest.raises(RouterError):
        manager.subscribe(
            _QUERY, operations=(Operation.d,), instructions="Updated instructions"
        )
    pending = repository.get(_QUERY)
    assert pending is not None and pending.status == "pending_update"
    assert pending.incarnation == original.incarnation
    assert pending.operations == (Operation.d,)
    assert pending.instructions == "Updated instructions"
    with pytest.raises(SubscriptionCommandError) as failure:
        _subscribe(manager)
    assert failure.value.category == "pending_operation"
    _confirm_subscribe(
        router, incarnation=original.incarnation, operations=(Operation.d,)
    )
    updated = manager.subscribe(
        _QUERY, operations=(Operation.d,), instructions="Updated instructions"
    )
    assert updated.status == "active"
    assert updated.incarnation == original.incarnation
    assert updated == repository.get(_QUERY)


@pytest.mark.parametrize("pending_status", ("pending_subscribe", "pending_update"))
def test_matching_pending_command_resumes_but_conflicting_commands_are_rejected(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    active_intent: SubscriptionIntent,
    pending_status: SubscriptionStatus,
) -> None:
    pending = active_intent.model_copy(update={"status": pending_status}, deep=True)
    _seed(repository, pending)
    with pytest.raises(SubscriptionCommandError) as changed:
        manager.subscribe(
            _QUERY, operations=pending.operations, instructions="Different operation"
        )
    assert changed.value.category == "pending_operation"
    with pytest.raises(SubscriptionCommandError) as removal:
        manager.unsubscribe(_QUERY)
    assert removal.value.category == "pending_operation"
    assert repository.get(_QUERY) == pending
    assert router.subscribe_requests == router.unsubscribe_requests == ()

    _confirm_subscribe(
        router, incarnation=pending.incarnation, operations=pending.operations
    )
    restored = manager.subscribe(
        _QUERY,
        operations=tuple(reversed(pending.operations)),
        instructions=pending.instructions,
    )
    assert restored.status == "active"
    assert restored.incarnation == pending.incarnation
    assert len(router.subscribe_requests) == 1


def test_subscribe_cannot_replace_pending_unsubscribe(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    active_intent: SubscriptionIntent,
) -> None:
    pending = active_intent.model_copy(
        update={"status": "pending_unsubscribe"}, deep=True
    )
    _seed(repository, pending)
    with pytest.raises(SubscriptionCommandError) as failure:
        _subscribe(manager)
    assert failure.value.category == "pending_operation"
    assert repository.get(_QUERY) == pending
    assert router.subscribe_requests == ()


@pytest.mark.parametrize("query_reappeared", (False, True))
@pytest.mark.parametrize("removed", (False, True))
def test_unavailable_intent_requires_durable_unsubscribe_then_fresh_incarnation(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    active_intent: SubscriptionIntent,
    catalog: ListQueriesResponse,
    empty_catalog: ListQueriesResponse,
    query_reappeared: bool,
    removed: bool,
) -> None:
    unavailable = active_intent.model_copy(update={"status": "unavailable"}, deep=True)
    _seed(repository, unavailable)
    manager.reconcile(catalog if query_reappeared else empty_catalog)
    assert repository.get(_QUERY) == unavailable
    with pytest.raises(
        SubscriptionCommandError, match="Unsubscribe first, then subscribe again"
    ) as failure:
        _subscribe(manager)
    assert failure.value.category == "unavailable"
    assert router.subscribe_requests == ()

    router.queue_unsubscribe_error("transport")
    with pytest.raises(RouterError):
        manager.unsubscribe(_QUERY)
    pending = repository.get(_QUERY)
    assert pending is not None and pending.status == "pending_unsubscribe"
    assert pending.incarnation == unavailable.incarnation
    _confirm_unsubscribe(router, removed=removed)
    manager.unsubscribe(_QUERY)
    assert repository.get(_QUERY) is None
    assert {
        request.subscription_incarnation for request in router.unsubscribe_requests
    } == {unavailable.incarnation}

    manager.reconcile(catalog)
    _confirm_subscribe(router)
    fresh = _subscribe(manager)
    assert fresh.incarnation != unavailable.incarnation
    assert fresh == repository.get(_QUERY)
    _confirm_subscribe(router, incarnation=fresh.incarnation, operations=(Operation.d,))
    updated = manager.subscribe(
        _QUERY, operations=(Operation.d,), instructions="An ordinary update"
    )
    assert updated.incarnation == fresh.incarnation
    snapshot = repository.load()
    assert snapshot is not None and set(snapshot.document.intents) == {_QUERY}


@pytest.mark.parametrize("status", ("active", "pending_subscribe", "pending_update"))
def test_restart_reasserts_saved_intent_without_changing_its_incarnation_or_catalog(
    repository: _Repository,
    router: _Router,
    catalog: ListQueriesResponse,
    active_intent: SubscriptionIntent,
    status: SubscriptionStatus,
) -> None:
    existing = active_intent.model_copy(update={"status": status}, deep=True)
    _seed(repository, existing)
    _confirm_subscribe(
        router, incarnation=existing.incarnation, operations=existing.operations
    )
    changed_catalog = parse(ListQueriesResponse, to_wire(catalog))
    changed_catalog.queries[0].title = "Updated operator description"
    manager = DrasiSubscriptionManager(repository=repository, router=router)
    manager.reconcile(changed_catalog)
    restored = repository.get(_QUERY)
    assert restored is not None and restored.status == "active"
    assert restored.incarnation == existing.incarnation
    assert restored.catalog_snapshot == existing.catalog_snapshot
    assert manager.list_subscriptions() == (restored,)
    assert router.closed is False


def test_restart_finishes_retired_unsubscribe_and_retains_other_retired_intent(
    repository: _Repository,
    router: _Router,
    empty_catalog: ListQueriesResponse,
    intent_document: IntentDocument,
) -> None:
    removal = intent_document.intents[_QUERY].model_copy(
        update={"status": "pending_unsubscribe"}, deep=True
    )
    other = intent_document.intents["rollout-status"]
    _seed(repository, other, removal)
    _confirm_unsubscribe(router, removed=False)
    manager = DrasiSubscriptionManager(repository=repository, router=router)
    manager.reconcile(empty_catalog)
    assert repository.get(_QUERY) is None
    retained = repository.get(other.query_id)
    assert retained is not None and retained.status == "unavailable"
    assert retained.incarnation == other.incarnation
    assert retained.catalog_snapshot == other.catalog_snapshot
    assert router.subscribe_requests == ()
    assert manager.list_subscriptions() == (retained,)


def test_reconciliation_removals_precede_failing_reassertions(
    repository: _Repository,
    router: _Router,
    catalog: ListQueriesResponse,
    intent_document: IntentDocument,
) -> None:
    active = intent_document.intents[_QUERY]
    removal = intent_document.intents["rollout-status"].model_copy(
        update={"status": "pending_unsubscribe"}, deep=True
    )
    _seed(repository, active, removal)
    _confirm_unsubscribe(router, query_id=removal.query_id)
    router.queue_subscribe_error("transport")
    manager = DrasiSubscriptionManager(repository=repository, router=router)
    with pytest.raises(RouterError):
        manager.reconcile(catalog)
    assert repository.get(removal.query_id) is None
    with pytest.raises(SubscriptionCommandError) as failure:
        _subscribe(manager)
    assert failure.value.category == "unavailable"


@pytest.mark.parametrize("stage", ("pending_subscribe", "active"))
@pytest.mark.parametrize("committed", (False, True))
def test_subscribe_storage_failures_remain_visible_and_recover_after_restart(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    catalog: ListQueriesResponse,
    stage: str,
    committed: bool,
) -> None:
    def fail_write(document: IntentDocument) -> None:
        if document.intents[_QUERY].status == stage:
            repository.before_save = repository.after_save = None
            raise IntentStoreError("unavailable")

    if committed:
        repository.after_save = fail_write
    else:
        repository.before_save = fail_write
    if stage == "active":
        _confirm_subscribe(router)
    with pytest.raises(IntentStoreError) as failure:
        _subscribe(manager)
    assert failure.value.category == "unavailable"
    assert len(router.subscribe_requests) == (1 if stage == "active" else 0)

    persisted = repository.get(_QUERY)
    fresh_router = _Router(repository.scope)
    restarted = DrasiSubscriptionManager(repository=repository, router=fresh_router)
    if persisted is not None:
        _confirm_subscribe(fresh_router, incarnation=persisted.incarnation)
    restarted.reconcile(catalog)
    if persisted is None:
        _confirm_subscribe(fresh_router, incarnation=_incarnation(2))
    recovered = _subscribe(restarted)
    assert recovered == repository.get(_QUERY)
    assert recovered.status == "active"
    if persisted is not None:
        assert recovered.incarnation == persisted.incarnation
    assert len(fresh_router.subscribe_requests) == 1


@pytest.mark.parametrize("stage", ("pending", "removal"))
@pytest.mark.parametrize("committed", (False, True))
def test_unsubscribe_storage_failures_do_not_claim_success_or_lose_recovery(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    catalog: ListQueriesResponse,
    stage: str,
    committed: bool,
) -> None:
    _confirm_subscribe(router)
    original = _subscribe(manager)

    def fail_write(document: IntentDocument) -> None:
        is_removal = _QUERY not in document.intents
        if is_removal == (stage == "removal"):
            repository.before_save = repository.after_save = None
            raise IntentStoreError("unavailable")

    if committed:
        repository.after_save = fail_write
    else:
        repository.before_save = fail_write
    if stage == "removal":
        _confirm_unsubscribe(router)
    with pytest.raises(IntentStoreError) as failure:
        manager.unsubscribe(_QUERY)
    assert failure.value.category == "unavailable"
    assert len(router.unsubscribe_requests) == (1 if stage == "removal" else 0)

    retained = repository.get(_QUERY)
    fresh_router = _Router(repository.scope)
    restarted = DrasiSubscriptionManager(repository=repository, router=fresh_router)
    if retained is not None:
        assert retained.incarnation == original.incarnation
        if retained.status == "pending_unsubscribe":
            _confirm_unsubscribe(fresh_router, removed=False)
        else:
            _confirm_subscribe(fresh_router, incarnation=retained.incarnation)
    restarted.reconcile(catalog)
    if repository.get(_QUERY) is not None:
        _confirm_unsubscribe(fresh_router)
    restarted.unsubscribe(_QUERY)
    assert repository.get(_QUERY) is None


def test_failure_to_persist_router_error_does_not_erase_pending_intent(
    manager: DrasiSubscriptionManager, repository: _Repository, router: _Router
) -> None:
    def fail_outcome(request: SubscribeRequest) -> None:
        repository.fail_next("save", "unavailable")

    router.before_subscribe = fail_outcome
    router.queue_subscribe_error("transport")
    with pytest.raises(IntentStoreError) as failure:
        _subscribe(manager)
    assert failure.value.category == "unavailable"
    pending = repository.get(_QUERY)
    assert pending is not None and pending.status == "pending_subscribe"
    assert pending.last_outcome is None


@pytest.mark.parametrize("stage", ("pending_subscribe", "active"))
def test_document_conflicts_merge_only_the_targeted_query_without_repeating_rpc(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    intent_document: IntentDocument,
    stage: str,
) -> None:
    other = intent_document.intents["rollout-status"]

    def concurrent_writer(document: IntentDocument) -> None:
        if document.intents[_QUERY].status != stage:
            return
        repository.before_save = None
        snapshot = repository.load()
        assert snapshot is not None
        snapshot.document.intents[other.query_id] = other
        InMemoryIntentRepository.save(
            repository, snapshot.document, expected_etag=snapshot.etag
        )

    repository.before_save = concurrent_writer
    _confirm_subscribe(router)
    result = _subscribe(manager)
    assert result == repository.get(_QUERY)
    assert repository.get(other.query_id) == other
    assert len(router.subscribe_requests) == 1
    assert repository.attempts == 3


def test_same_query_conflict_does_not_overwrite_another_transition(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    active_intent: SubscriptionIntent,
) -> None:
    competing = active_intent.model_copy(
        update={"status": "pending_unsubscribe"}, deep=True
    )

    def replace_query(request: SubscribeRequest) -> None:
        snapshot = repository.load()
        assert snapshot is not None
        snapshot.document.intents[_QUERY] = competing
        InMemoryIntentRepository.save(
            repository, snapshot.document, expected_etag=snapshot.etag
        )

    router.before_subscribe = replace_query
    _confirm_subscribe(router)
    with pytest.raises(IntentStoreError) as failure:
        _subscribe(manager)
    assert failure.value.category == "conflict"
    assert repository.get(_QUERY) == competing
    assert len(router.subscribe_requests) == 1


def test_conflict_retry_budget_is_bounded_and_never_initializes_over_state(
    manager: DrasiSubscriptionManager, repository: _Repository, router: _Router
) -> None:
    for _ in range(10):
        repository.fail_next("save", "conflict")
    with pytest.raises(IntentStoreError) as failure:
        _subscribe(manager)
    assert failure.value.category == "conflict"
    assert repository.attempts == 10
    assert repository.get(_QUERY) is None
    assert router.subscribe_requests == ()


def test_same_query_commands_are_serialized_while_listing_can_observe_pending(
    manager: DrasiSubscriptionManager, repository: _Repository, router: _Router
) -> None:
    entered, release, second_started = Event(), Event(), Event()

    def wait_at_router(request: SubscribeRequest) -> None:
        entered.set()
        assert release.wait(5)

    def repeat() -> SubscriptionIntent:
        second_started.set()
        return _subscribe(manager)

    router.before_subscribe = wait_at_router
    _confirm_subscribe(router)
    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(_subscribe, manager)
        try:
            assert entered.wait(5)
            second = executor.submit(repeat)
            assert second_started.wait(5)
            assert not second.done()
            listing = executor.submit(manager.list_subscriptions).result(timeout=5)
            assert listing[0].status == "pending_subscribe"
        finally:
            release.set()
        assert first.result(timeout=5) == second.result(timeout=5)
    assert len(router.subscribe_requests) == 1
    assert repository.attempts == 2


def test_catalog_and_returned_intents_are_detached(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    catalog: ListQueriesResponse,
) -> None:
    expected = to_wire(catalog)
    catalog.queries.clear()
    _confirm_subscribe(router)
    returned = _subscribe(manager)
    returned.instructions = "Caller mutation"
    returned.catalog_snapshot.queries.clear()
    listed = manager.list_subscriptions()
    assert to_wire(listed[0].catalog_snapshot) == expected
    assert listed[0].instructions == _INSTRUCTIONS
    listed[0].catalog_snapshot.queries.clear()
    assert manager.list_subscriptions()[0].catalog_snapshot.queries
    assert repository.get(_QUERY) != returned


@pytest.mark.parametrize(
    ("operations", "instructions"),
    (
        ((), _INSTRUCTIONS),
        ((Operation.i, Operation.i), _INSTRUCTIONS),
        (("unsupported",), _INSTRUCTIONS),
        (_OPERATIONS, ""),
        (_OPERATIONS, " \t"),
        (_OPERATIONS, {"private": _INSTRUCTIONS}),
    ),
)
def test_invalid_commands_do_not_write_or_expose_instructions(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    operations: tuple[Operation, ...],
    instructions: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(SubscriptionCommandError) as failure:
        manager.subscribe(_QUERY, operations=operations, instructions=instructions)
    assert failure.value.category == "invalid_input"
    assert repository.saved == []
    assert router.subscribe_requests == ()
    assert _INSTRUCTIONS not in caplog.text
    assert _INSTRUCTIONS not in "".join(traceback.format_exception(failure.value))


@pytest.mark.parametrize("query_id", ("", None))
def test_invalid_query_ids_are_not_silent_unsubscribe_success(
    manager: DrasiSubscriptionManager, query_id: str
) -> None:
    with pytest.raises(SubscriptionCommandError) as failure:
        manager.unsubscribe(query_id)
    assert failure.value.category == "invalid_input"


def test_empty_catalog_is_valid_but_unknown_queries_are_rejected(
    repository: _Repository, router: _Router, empty_catalog: ListQueriesResponse
) -> None:
    manager = DrasiSubscriptionManager(repository=repository, router=router)
    manager.reconcile(empty_catalog)
    assert manager.list_subscriptions() == ()
    with pytest.raises(SubscriptionCommandError) as failure:
        _subscribe(manager)
    assert failure.value.category == "unknown_query"
    assert repository.saved == []
    assert router.subscribe_requests == ()


@pytest.mark.parametrize("category", ("unavailable", "corrupt", "unsupported_version"))
def test_store_failures_never_become_empty_subscription_listings(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    category: StoreErrorCategory,
) -> None:
    repository.fail_next("load", category)
    with pytest.raises(IntentStoreError) as failure:
        manager.list_subscriptions()
    assert failure.value.category == category


def test_missing_repository_requires_preparation_not_implicit_initialization(
    scope: SubscriptionScope, router: _Router, catalog: ListQueriesResponse
) -> None:
    repository = InMemoryIntentRepository(scope)
    manager = DrasiSubscriptionManager(repository=repository, router=router)
    with pytest.raises(IntentStoreError) as failure:
        manager.reconcile(catalog)
    assert failure.value.category == "unavailable"
    assert repository.load() is None


@pytest.mark.parametrize("defect", ("scope", "etag"))
def test_invalid_repository_snapshots_are_not_treated_as_empty_intent(
    manager: DrasiSubscriptionManager,
    repository: _Repository,
    router: _Router,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    snapshot = repository.load()
    assert snapshot is not None
    document = snapshot.document
    if defect == "scope":
        document.scope = repository.scope.model_copy(update={"app_id": "another-app"})
    broken = IntentSnapshot(
        document=document, etag="" if defect == "etag" else snapshot.etag
    )
    monkeypatch.setattr(repository, "load", lambda: broken)
    with pytest.raises(IntentStoreError) as failure:
        manager.list_subscriptions()
    assert failure.value.category == ("corrupt" if defect == "scope" else "unavailable")
    assert repository.saved == []
    assert router.subscribe_requests == router.unsubscribe_requests == ()


def test_scope_mismatch_and_invalid_catalog_fail_before_mutation(
    repository: _Repository,
    router: _Router,
    catalog: ListQueriesResponse,
    scope: SubscriptionScope,
) -> None:
    other_scope = scope.model_copy(update={"app_id": "another-app"})
    with pytest.raises(ValueError, match="scopes"):
        DrasiSubscriptionManager(repository=repository, router=_Router(other_scope))
    manager = DrasiSubscriptionManager(repository=repository, router=router)
    wrong_catalog = catalog.model_copy(update={"router_id": "other/router"}, deep=True)
    with pytest.raises(RouterError) as failure:
        manager.reconcile(wrong_catalog)
    assert failure.value.category == "invalid_response"
    assert repository.saved == []
    assert router.subscribe_requests == router.unsubscribe_requests == ()


def test_invalid_router_lifecycle_is_not_disguised_as_success(
    manager: DrasiSubscriptionManager, router: _Router, repository: _Repository
) -> None:
    router.close()
    with pytest.raises(RuntimeError, match="closed"):
        _subscribe(manager)
    pending = repository.get(_QUERY)
    assert pending is not None and pending.status == "pending_subscribe"
