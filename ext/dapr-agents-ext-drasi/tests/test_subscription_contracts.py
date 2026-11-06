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

"""Focused behavioral tests for subscription boundaries and shared fakes."""

from __future__ import annotations

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

from dapr_agents.ext.drasi._interfaces import (
    IntentReader,
    IntentRepository,
    IntentStoreError,
    RouterClient,
    RouterError,
    SubscriptionCommandError,
    SubscriptionManager,
)
from dapr_agents.ext.drasi._models import (
    IntentDocument,
    MutationOutcome,
    RouterFailureCategory,
    SubscriptionIntent,
    SubscriptionScope,
    SubscriptionStatus,
)

from .fakes import (
    InMemoryIntentRepository,
    ScriptedRouterClient,
    ScriptedSubscriptionManager,
)


def _replace_intent(
    document: IntentDocument,
    query_id: str,
    *,
    instructions: str,
) -> IntentDocument:
    replacement = document.intents[query_id].model_copy(
        update={"instructions": instructions},
        deep=True,
    )
    intents = {
        key: value.model_copy(deep=True) for key, value in document.intents.items()
    }
    intents[query_id] = replacement
    return IntentDocument(
        format_version=1,
        scope=document.scope,
        intents=intents,
    )


def _subscribe_request(
    scope: SubscriptionScope,
    *,
    operations: tuple[Operation, ...] = (Operation.i, Operation.u),
) -> SubscribeRequest:
    return parse(
        SubscribeRequest,
        {
            "query_id": "service-errors",
            "operations": [operation.value for operation in operations],
            "subscriber": to_wire(scope.subscriber),
            "subscription_incarnation": "incarnation-service-errors",
        },
    )


def _unsubscribe_request(scope: SubscriptionScope) -> UnsubscribeRequest:
    return parse(
        UnsubscribeRequest,
        {
            "query_id": "service-errors",
            "subscriber": to_wire(scope.subscriber),
            "subscription_incarnation": "incarnation-service-errors",
        },
    )


def _updated_response(scope: SubscriptionScope) -> SubscribeResponse:
    return parse(
        SubscribeResponse,
        {
            "query_id": "service-errors",
            "operations": ["i", "u"],
            "subscription_incarnation": "incarnation-service-errors",
            "topic_name": scope.inbox_topic,
            "status": "updated",
        },
    )


def test_repository_detaches_initialize_load_get_and_save(
    scope: SubscriptionScope,
    intent_document: IntentDocument,
) -> None:
    repository = InMemoryIntentRepository(scope)
    original_instruction = intent_document.intents["service-errors"].instructions

    repository.initialize(intent_document)
    intent_document.intents["service-errors"].instructions = "caller mutation"

    loaded = repository.load()
    assert loaded is not None
    assert (
        loaded.document.intents["service-errors"].instructions == original_instruction
    )
    loaded_wire = loaded.document.model_dump(mode="json")
    assert (
        "usage"
        not in loaded_wire["intents"]["service-errors"]["catalog_snapshot"]["queries"][
            1
        ]
    )

    loaded.document.intents["service-errors"].instructions = "snapshot mutation"
    fetched = repository.get("service-errors")
    assert fetched is not None
    assert fetched.instructions == original_instruction

    replacement = _replace_intent(
        loaded.document,
        "service-errors",
        instructions="persisted replacement",
    )
    result = repository.save(replacement, expected_etag=loaded.etag)
    replacement.intents["service-errors"].instructions = "post-save mutation"

    assert result is None
    reread = repository.load()
    assert reread is not None
    assert reread.etag != loaded.etag
    assert (
        reread.document.intents["service-errors"].instructions
        == "persisted replacement"
    )


def test_whole_document_conflict_requires_reread_and_explicit_merge(
    scope: SubscriptionScope,
    intent_document: IntentDocument,
) -> None:
    repository = InMemoryIntentRepository(scope)
    repository.initialize(intent_document)
    first = repository.load()
    second = repository.load()
    assert first is not None
    assert second is not None
    assert first.etag == second.etag

    first_update = _replace_intent(
        first.document,
        "service-errors",
        instructions="first writer changed service errors",
    )
    repository.save(first_update, expected_etag=first.etag)

    stale_second_update = _replace_intent(
        second.document,
        "rollout-status",
        instructions="second writer changed rollout status",
    )
    with pytest.raises(IntentStoreError) as conflict:
        repository.save(stale_second_update, expected_etag=second.etag)
    assert conflict.value.category == "conflict"

    after_conflict = repository.load()
    assert after_conflict is not None
    assert (
        after_conflict.document.intents["service-errors"].instructions
        == "first writer changed service errors"
    )
    assert (
        after_conflict.document.intents["rollout-status"].instructions
        == intent_document.intents["rollout-status"].instructions
    )

    merged = _replace_intent(
        after_conflict.document,
        "rollout-status",
        instructions="second writer changed rollout status",
    )
    repository.save(merged, expected_etag=after_conflict.etag)

    final = repository.load()
    assert final is not None
    assert (
        final.document.intents["service-errors"].instructions
        == "first writer changed service errors"
    )
    assert (
        final.document.intents["rollout-status"].instructions
        == "second writer changed rollout status"
    )


def test_repository_detaches_nested_snapshot_contents(
    scope: SubscriptionScope,
    intent_document: IntentDocument,
) -> None:
    repository = InMemoryIntentRepository(scope)
    repository.initialize(intent_document)
    original_title = (
        intent_document.intents["service-errors"].catalog_snapshot.queries[0].title
    )
    snapshot = repository.load()
    assert snapshot is not None
    snapshot.document.intents["service-errors"].catalog_snapshot.queries.clear()

    intent = repository.get("service-errors")
    assert intent is not None
    assert intent.catalog_snapshot.queries[0].title == original_title
    intent.catalog_snapshot.queries[0].title = "reader mutation"

    reread = repository.get("service-errors")
    assert reread is not None
    assert reread.catalog_snapshot.queries[0].title == original_title


def test_initialize_is_unconditional_not_create_if_absent(
    scope: SubscriptionScope,
    intent_document: IntentDocument,
) -> None:
    repository = InMemoryIntentRepository(scope)
    repository.initialize(intent_document)
    first = repository.load()
    assert first is not None

    replacement = _replace_intent(
        intent_document,
        "service-errors",
        instructions="exclusive owner reinitialized the document",
    )
    repository.initialize(replacement)

    second = repository.load()
    assert second is not None
    assert second.etag != first.etag
    assert (
        second.document.intents["service-errors"].instructions
        == "exclusive owner reinitialized the document"
    )


def test_repository_missing_state_is_distinct_from_failures(
    scope: SubscriptionScope,
) -> None:
    repository = InMemoryIntentRepository(scope)

    assert repository.load() is None
    assert repository.get("service-errors") is None

    for operation in ("load", "get"):
        for category in ("unavailable", "corrupt", "unsupported_version"):
            repository.fail_next(operation, category)
            with pytest.raises(IntentStoreError) as failure:
                if operation == "load":
                    repository.load()
                else:
                    repository.get("service-errors")
            assert failure.value.category == category


def test_repository_rejects_empty_or_stale_etags_without_mutation(
    scope: SubscriptionScope,
    intent_document: IntentDocument,
) -> None:
    repository = InMemoryIntentRepository(scope)
    repository.initialize(intent_document)
    snapshot = repository.load()
    assert snapshot is not None
    update = _replace_intent(
        snapshot.document,
        "service-errors",
        instructions="candidate replacement",
    )

    with pytest.raises(ValueError, match="nonempty"):
        repository.save(update, expected_etag="")
    with pytest.raises(IntentStoreError) as conflict:
        repository.save(update, expected_etag="stale")
    assert conflict.value.category == "conflict"

    unchanged = repository.load()
    assert unchanged is not None
    assert (
        unchanged.document.intents["service-errors"].instructions
        != "candidate replacement"
    )


def test_scripted_router_detaches_catalogs_requests_and_responses(
    scope: SubscriptionScope,
    catalog: ListQueriesResponse,
) -> None:
    router = ScriptedRouterClient(scope)
    router.queue_catalog(catalog)
    router.queue_catalog(catalog)
    catalog.queries[0].title = "caller mutation"

    first = router.list_queries()
    first.queries[0].title = "returned mutation"
    second = router.list_queries()
    assert second.queries[0].title == "Service errors"

    request = _subscribe_request(scope)
    response = _updated_response(scope)
    router.queue_subscribe(response)
    response.topic_name = "caller mutation"
    returned = router.subscribe(request)
    request.subscription_incarnation = "caller mutation"
    assert returned.topic_name == scope.inbox_topic
    returned.topic_name = "returned mutation"

    assert (
        router.subscribe_requests[0].subscription_incarnation
        == "incarnation-service-errors"
    )
    assert router.scope.inbox_topic in to_wire(_updated_response(scope)).values()

    recorded = router.subscribe_requests[0]
    recorded.subscription_incarnation = "property mutation"
    assert (
        router.subscribe_requests[0].subscription_incarnation
        == "incarnation-service-errors"
    )


def test_scripted_manager_detaches_results_lists_and_reconciled_catalogs(
    active_intent: SubscriptionIntent,
    catalog: ListQueriesResponse,
) -> None:
    manager = ScriptedSubscriptionManager()
    manager.queue_subscribe(active_intent)
    active_intent.instructions = "caller mutation"

    returned = manager.subscribe(
        "service-errors",
        operations=(Operation.i, Operation.u),
        instructions="Investigate service errors.",
    )
    returned.instructions = "returned mutation"
    assert manager.subscribe_calls[0].operations == (Operation.i, Operation.u)

    manager.set_subscriptions((returned,))
    returned.instructions = "post-list-setup mutation"
    first_list = manager.list_subscriptions()
    first_list[0].instructions = "list mutation"
    assert manager.list_subscriptions()[0].instructions == "returned mutation"

    manager.queue_reconcile()
    manager.reconcile(catalog)
    catalog.queries[0].title = "caller mutation"
    reconciled = manager.reconciled_catalogs[0]
    assert reconciled.queries[0].title == "Service errors"
    reconciled.queries[0].title = "property mutation"
    assert manager.reconciled_catalogs[0].queries[0].title == "Service errors"


def test_manager_lists_active_pending_and_unavailable_snapshots(
    status_intents: dict[SubscriptionStatus, SubscriptionIntent],
) -> None:
    manager = ScriptedSubscriptionManager()
    manager.set_subscriptions(tuple(status_intents.values()))

    listed = manager.list_subscriptions()
    assert {intent.status for intent in listed} == {
        "pending_subscribe",
        "pending_update",
        "active",
        "pending_unsubscribe",
        "unavailable",
    }

    listed[0].status = "active"
    assert {intent.status for intent in manager.list_subscriptions()} == {
        "pending_subscribe",
        "pending_update",
        "active",
        "pending_unsubscribe",
        "unavailable",
    }


def test_manager_listing_requires_setup_and_preserves_scripted_failures() -> None:
    manager = ScriptedSubscriptionManager()

    with pytest.raises(AssertionError, match="subscription listing"):
        manager.list_subscriptions()

    manager.set_subscriptions(())
    errors = (
        IntentStoreError("unavailable"),
        SubscriptionCommandError("unavailable"),
    )
    for error in errors:
        manager.queue_list_error(error)
        with pytest.raises(type(error)) as failure:
            manager.list_subscriptions()
        assert failure.value is error

    assert manager.list_subscriptions() == ()


@pytest.mark.parametrize(
    ("category", "expected_outcome"),
    (
        ("state_unavailable", "uncertain"),
        ("transport", "uncertain"),
        ("invalid_response", "uncertain"),
        ("invalid_arguments", "rejected"),
        ("unknown_query", "rejected"),
        ("incarnation_conflict", "rejected"),
    ),
)
def test_scripted_router_preserves_typed_mutation_failures(
    scope: SubscriptionScope,
    category: RouterFailureCategory,
    expected_outcome: MutationOutcome,
) -> None:
    router = ScriptedRouterClient(scope)
    router.queue_subscribe_error(category)

    with pytest.raises(RouterError) as failure:
        router.subscribe(_subscribe_request(scope))
    assert failure.value.category == category
    assert failure.value.mutation_outcome == expected_outcome


def test_router_retry_script_reuses_request_and_handles_confirmed_absence(
    scope: SubscriptionScope,
) -> None:
    router = ScriptedRouterClient(scope)
    subscribe_request = _subscribe_request(scope)
    router.queue_subscribe_error("transport")
    router.queue_subscribe(_updated_response(scope))

    with pytest.raises(RouterError) as subscribe_failure:
        router.subscribe(subscribe_request)
    assert subscribe_failure.value.mutation_outcome == "uncertain"
    response = router.subscribe(subscribe_request)
    assert response.status.value == "updated"
    assert [
        request.subscription_incarnation for request in router.subscribe_requests
    ] == ["incarnation-service-errors", "incarnation-service-errors"]
    assert to_wire(router.subscribe_requests[0]) == to_wire(
        router.subscribe_requests[1]
    )

    unsubscribe_request = _unsubscribe_request(scope)
    router.queue_unsubscribe_error("invalid_response")
    router.queue_unsubscribe(
        parse(
            UnsubscribeResponse,
            {"query_id": "service-errors", "removed": False},
        )
    )

    with pytest.raises(RouterError) as unsubscribe_failure:
        router.unsubscribe(unsubscribe_request)
    assert unsubscribe_failure.value.mutation_outcome == "uncertain"
    unsubscribe_response = router.unsubscribe(unsubscribe_request)
    assert unsubscribe_response.removed is False
    assert to_wire(router.unsubscribe_requests[0]) == to_wire(
        router.unsubscribe_requests[1]
    )


def test_unconfigured_fake_mutations_fail_loudly(
    scope: SubscriptionScope,
    active_intent: SubscriptionIntent,
    catalog: ListQueriesResponse,
) -> None:
    router = ScriptedRouterClient(scope)
    manager = ScriptedSubscriptionManager()

    with pytest.raises(AssertionError, match="subscribe result"):
        router.subscribe(_subscribe_request(scope))
    with pytest.raises(AssertionError, match="manager subscribe"):
        manager.subscribe(
            active_intent.query_id,
            operations=active_intent.operations,
            instructions=active_intent.instructions,
        )
    with pytest.raises(AssertionError, match="unsubscribe result"):
        router.unsubscribe(_unsubscribe_request(scope))
    with pytest.raises(AssertionError, match="manager unsubscribe"):
        manager.unsubscribe(active_intent.query_id)
    with pytest.raises(AssertionError, match="manager reconcile"):
        manager.reconcile(catalog)


def test_scripted_manager_preserves_typed_errors(
    active_intent: SubscriptionIntent,
) -> None:
    manager = ScriptedSubscriptionManager()
    error = SubscriptionCommandError("unknown_query")
    manager.queue_subscribe_error(error)

    with pytest.raises(SubscriptionCommandError) as failure:
        manager.subscribe(
            active_intent.query_id,
            operations=active_intent.operations,
            instructions=active_intent.instructions,
        )
    assert failure.value is error


def _query_decision(manager: SubscriptionManager) -> SubscriptionIntent:
    return manager.subscribe(
        "service-errors",
        operations=(Operation.i, Operation.u),
        instructions="Investigate service errors.",
    )


def _router_mutation(
    router: RouterClient,
    intent: SubscriptionIntent,
) -> SubscribeResponse:
    request = parse(
        SubscribeRequest,
        {
            "query_id": intent.query_id,
            "operations": [operation.value for operation in intent.operations],
            "subscriber": to_wire(router.scope.subscriber),
            "subscription_incarnation": intent.incarnation,
        },
    )
    assert "instructions" not in to_wire(request)
    return router.subscribe(request)


def _read_intent(reader: IntentReader, query_id: str) -> SubscriptionIntent | None:
    return reader.get(query_id)


def test_protocol_consumers_need_only_their_narrow_boundaries(
    scope: SubscriptionScope,
    active_intent: SubscriptionIntent,
    intent_document: IntentDocument,
) -> None:
    manager_fake = ScriptedSubscriptionManager()
    manager_fake.queue_subscribe(active_intent)
    manager: SubscriptionManager = manager_fake
    decided = _query_decision(manager)
    assert decided.query_id == "service-errors"

    router_fake = ScriptedRouterClient(scope)
    router_fake.queue_subscribe(_updated_response(scope))
    router: RouterClient = router_fake
    assert _router_mutation(router, decided).status.value == "updated"

    repository_fake = InMemoryIntentRepository(scope)
    repository_fake.initialize(intent_document)
    repository: IntentRepository = repository_fake
    reader: IntentReader = repository
    read = _read_intent(reader, "service-errors")
    assert read is not None
    assert read.query_id == "service-errors"


def test_fake_close_is_idempotent_and_rejects_post_close_use(
    scope: SubscriptionScope,
    catalog: ListQueriesResponse,
) -> None:
    router_fake = ScriptedRouterClient(scope)
    router_fake.queue_catalog(catalog)
    router_fake.queue_subscribe(_updated_response(scope))
    router_fake.queue_unsubscribe(
        parse(
            UnsubscribeResponse,
            {"query_id": "service-errors", "removed": True},
        )
    )
    router: RouterClient = router_fake
    router.close()
    router.close()

    assert router_fake.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        router.list_queries()
    with pytest.raises(RuntimeError, match="closed"):
        router.subscribe(_subscribe_request(scope))
    with pytest.raises(RuntimeError, match="closed"):
        router.unsubscribe(_unsubscribe_request(scope))
