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

"""Focused validation tests for Drasi subscription contract values."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
from drasi_agent_router_contracts import AgentDelivery, ListQueriesResponse, parse
from drasi_agent_router_contracts.models.Operation import Operation
from pydantic import ValidationError

from dapr_agents.ext.drasi._models import (
    Discard,
    IntentDocument,
    Poison,
    ResolvedDrasiConfig,
    Retry,
    RouterFailureCategory,
    MutationOutcome,
    RouterOperationOutcome,
    SchedulingInput,
    SubscriptionIntent,
    SubscriptionScope,
    SubscriptionStatus,
)


def test_scope_uses_exact_shared_identity_topics_and_fresh_subscribers(
    scope: SubscriptionScope,
) -> None:
    assert (
        scope.inbox_topic
        == "drasi-ai1-6yqocy32t35jddui3sl5mkfsse7eeigdr5vklfeskji77pe4hr2a"
    )
    assert (
        scope.dead_letter_topic
        == "drasi-ad1-6yqocy32t35jddui3sl5mkfsse7eeigdr5vklfeskji77pe4hr2a"
    )

    subscriber = scope.subscriber
    subscriber.agent_name = "mutated"

    assert scope.agent_name == "CheckoutSRE"
    assert scope.subscriber.agent_name == "CheckoutSRE"
    with pytest.raises(ValidationError):
        scope.agent_name = "mutated"


def test_resolved_config_is_exact_immutable_and_forbids_invalid_values(
    scope: SubscriptionScope,
) -> None:
    config = ResolvedDrasiConfig(
        scope=scope,
        pubsub_name="drasi-pubsub",
        state_store_name="agent-state",
        workflow_name="AgentWorkflow",
        dapr_http_port=3500,
    )

    assert config.router_mcp_url == (
        "http://localhost:3500/v1.0/invoke/sre-router-reaction.drasi-system/method/mcp"
    )
    with pytest.raises(ValidationError):
        config.dapr_http_port = 3600

    invalid_documents = (
        {
            **config.model_dump(),
            "extra": True,
        },
        {
            **config.model_dump(),
            "pubsub_name": " ",
        },
        {
            **config.model_dump(),
            "dapr_http_port": 0,
        },
    )
    for document in invalid_documents:
        with pytest.raises(ValidationError):
            ResolvedDrasiConfig.model_validate(document)


@pytest.mark.parametrize(
    "document",
    (
        {
            "router_id": "missing-slash",
            "namespace": "applications",
            "app_id": "checkout-sre",
            "agent_name": "CheckoutSRE",
        },
        {
            "router_id": "drasi-system/router",
            "namespace": "",
            "app_id": "checkout-sre",
            "agent_name": "CheckoutSRE",
        },
        {
            "router_id": "drasi-system/router",
            "namespace": "applications",
            "app_id": "checkout sre",
            "agent_name": "CheckoutSRE",
        },
    ),
)
def test_scope_rejects_invalid_shared_identities(document: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        SubscriptionScope.model_validate(document)


def test_current_empty_catalog_is_valid(
    empty_catalog: ListQueriesResponse,
    scope: SubscriptionScope,
) -> None:
    assert empty_catalog.router_id == scope.router_id
    assert empty_catalog.queries == []


def test_intent_document_json_roundtrip_preserves_wire_omission_and_fields(
    intent_document: IntentDocument,
) -> None:
    wire_document = json.loads(intent_document.model_dump_json())
    service_query, rollout_query = wire_document["intents"]["service-errors"][
        "catalog_snapshot"
    ]["queries"]

    assert service_query["usage"] == "Select inserts to monitor newly matching errors."
    assert "usage" not in rollout_query

    restored = IntentDocument.model_validate_json(intent_document.model_dump_json())
    assert restored == intent_document
    assert restored.scope == intent_document.scope
    assert set(restored.intents) == {"service-errors", "rollout-status"}
    assert restored.intents["service-errors"].incarnation
    assert restored.intents["rollout-status"].last_outcome is None


def test_catalog_explicit_null_optional_field_is_invalid(
    active_intent: SubscriptionIntent,
) -> None:
    document = active_intent.model_dump(mode="json")
    queries = document["catalog_snapshot"]["queries"]
    queries[1]["usage"] = None

    with pytest.raises(ValidationError, match="Invalid router catalog snapshot"):
        SubscriptionIntent.model_validate(document)


def test_intent_and_document_detach_nested_catalog_inputs(
    scope: SubscriptionScope,
    catalog: ListQueriesResponse,
    active_intent: SubscriptionIntent,
) -> None:
    original_title = active_intent.catalog_snapshot.queries[0].title
    catalog.queries[0].title = "catalog caller mutation"
    assert active_intent.catalog_snapshot.queries[0].title == original_title

    document = IntentDocument(
        format_version=1,
        scope=scope,
        intents={active_intent.query_id: active_intent},
    )
    active_intent.catalog_snapshot.queries[0].title = "intent caller mutation"

    stored_intent = document.intents[active_intent.query_id]
    assert stored_intent.catalog_snapshot.queries[0].title == original_title
    assert IntentDocument.model_validate_json(document.model_dump_json()) == document


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("operations", ()),
        ("operations", (Operation.i, Operation.i)),
        ("operations", ("x",)),
        ("instructions", " \t"),
    ),
)
def test_intent_rejects_invalid_query_decisions(
    active_intent: SubscriptionIntent,
    field: str,
    value: object,
) -> None:
    document = active_intent.model_dump(mode="json")
    document[field] = value

    with pytest.raises(ValidationError):
        SubscriptionIntent.model_validate(document)


def test_intent_requires_all_fields(active_intent: SubscriptionIntent) -> None:
    document = active_intent.model_dump(mode="json")

    for required in (
        "query_id",
        "operations",
        "instructions",
        "catalog_snapshot",
        "incarnation",
        "status",
    ):
        incomplete = dict(document)
        incomplete.pop(required)
        with pytest.raises(ValidationError):
            SubscriptionIntent.model_validate(incomplete)


@pytest.mark.parametrize("version", (2, True, "1", None))
def test_document_rejects_unsupported_versions(
    intent_document: IntentDocument,
    version: object,
) -> None:
    document = intent_document.model_dump(mode="json")
    document["format_version"] = version

    with pytest.raises(ValidationError, match="Unsupported intent document"):
        IntentDocument.model_validate(document)


def test_document_requires_format_version(
    intent_document: IntentDocument,
) -> None:
    for required in ("format_version", "scope", "intents"):
        document = intent_document.model_dump(mode="json")
        document.pop(required)
        with pytest.raises(ValidationError):
            IntentDocument.model_validate(document)


def test_intent_and_document_reject_catalog_identity_mismatches(
    active_intent: SubscriptionIntent,
    intent_document: IntentDocument,
) -> None:
    absent_query = active_intent.model_dump(mode="json")
    absent_query["query_id"] = "missing-query"
    with pytest.raises(ValidationError, match="missing from its stored catalog"):
        SubscriptionIntent.model_validate(absent_query)

    wrong_key = intent_document.model_dump(mode="json")
    wrong_key["intents"]["wrong-key"] = wrong_key["intents"].pop("service-errors")
    with pytest.raises(ValidationError, match="key does not match"):
        IntentDocument.model_validate(wrong_key)

    wrong_scope = intent_document.model_dump(mode="json")
    wrong_scope["scope"]["router_id"] = "other/router"
    with pytest.raises(ValidationError, match="different router"):
        IntentDocument.model_validate(wrong_scope)

    duplicate_catalog = active_intent.model_dump(mode="json")
    duplicate_catalog["catalog_snapshot"]["queries"].append(
        duplicate_catalog["catalog_snapshot"]["queries"][0]
    )
    with pytest.raises(ValidationError, match="Invalid router catalog snapshot"):
        SubscriptionIntent.model_validate(duplicate_catalog)


def test_historical_catalog_remains_valid_for_retired_intent(
    scope: SubscriptionScope,
    active_intent: SubscriptionIntent,
    empty_catalog: ListQueriesResponse,
) -> None:
    assert empty_catalog.queries == []

    retired = active_intent.model_copy(update={"status": "unavailable"}, deep=True)
    document = IntentDocument(
        format_version=1,
        scope=scope,
        intents={retired.query_id: retired},
    )

    assert document.intents["service-errors"].catalog_snapshot.queries
    assert document.intents["service-errors"].status == "unavailable"


@pytest.mark.parametrize(
    ("category", "outcome"),
    (
        ("invalid_arguments", "rejected"),
        ("unknown_query", "rejected"),
        ("incarnation_conflict", "rejected"),
        ("state_unavailable", "uncertain"),
        ("transport", "uncertain"),
        ("invalid_response", "uncertain"),
    ),
)
def test_router_operation_outcome_enforces_category_consistency(
    category: RouterFailureCategory,
    outcome: MutationOutcome,
) -> None:
    result = RouterOperationOutcome(
        operation="subscribe",
        outcome=outcome,
        error_category=category,
    )
    assert result.outcome == outcome

    opposite = "confirmed" if outcome != "confirmed" else "uncertain"
    with pytest.raises(ValidationError, match="does not match"):
        RouterOperationOutcome(
            operation="subscribe",
            outcome=opposite,
            error_category=category,
        )


def test_confirmed_router_outcome_has_no_error_category() -> None:
    outcome = RouterOperationOutcome(operation="unsubscribe", outcome="confirmed")
    assert outcome.error_category is None

    with pytest.raises(ValidationError, match="does not match"):
        RouterOperationOutcome(
            operation="unsubscribe",
            outcome="uncertain",
        )


@pytest.mark.parametrize(
    "result",
    (
        Discard(reason="no_intent"),
        Discard(reason="unavailable"),
        Retry(reason="pending_subscription"),
        Retry(reason="state_corrupt"),
        Poison(reason="invalid_delivery"),
        Poison(reason="router_mismatch"),
    ),
)
def test_admission_decision_variants_are_plain_immutable_values(
    result: Discard | Retry | Poison,
) -> None:
    assert result.reason
    with pytest.raises(FrozenInstanceError):
        result.reason = result.reason


def test_scheduling_input_owns_immutable_hidden_task_text() -> None:
    task_text = "Investigate error 123 using the retained instructions."
    result = SchedulingInput(
        instance_id="scope:incarnation:event",
        event_id="drasi:v1:service-errors:42:i:0",
        task=task_text,
    )

    task_text += " caller mutation"
    assert result.task == "Investigate error 123 using the retained instructions."
    assert result.task not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.task = "replacement"


def test_intent_repr_excludes_raw_instructions(
    active_intent: SubscriptionIntent,
) -> None:
    assert active_intent.instructions not in repr(active_intent)


def test_published_canonical_deliveries_cover_insert_update_delete(
    canonical_delivery: AgentDelivery,
) -> None:
    assert canonical_delivery.event.op in {"i", "u", "d"}
    assert canonical_delivery.event.payload.source.queryId == "service-errors"


def test_current_lifecycle_fixtures_share_subscription_incarnation(
    active_intent: SubscriptionIntent,
    status_intents: dict[SubscriptionStatus, SubscriptionIntent],
    insert_delivery: AgentDelivery,
    update_delivery: AgentDelivery,
    delete_delivery: AgentDelivery,
) -> None:
    incarnations = {
        active_intent.incarnation,
        *(intent.incarnation for intent in status_intents.values()),
        insert_delivery.subscriptionIncarnation,
        update_delivery.subscriptionIncarnation,
        delete_delivery.subscriptionIncarnation,
    }

    assert incarnations == {"incarnation-service-errors"}
