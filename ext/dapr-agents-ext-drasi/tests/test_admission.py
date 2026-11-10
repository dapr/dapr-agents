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

"""Admission behavior using the frozen intent-reader boundary, without a runtime."""

from __future__ import annotations

import json
import logging
import sys
from importlib.resources import files
from typing import Any

import pytest
from drasi_agent_router_contracts import AgentDelivery, parse, to_wire
from drasi_agent_router_contracts.models.Operation import Operation
from pytest_mock import MockerFixture

from dapr_agents.ext.drasi._interfaces import AdmissionHandler
from dapr_agents.ext.drasi._models import (
    Discard,
    IntentDocument,
    Poison,
    Retry,
    SchedulingInput,
    SubscriptionIntent,
    SubscriptionScope,
    SubscriptionStatus,
)
from dapr_agents.ext.drasi.admission import DrasiAdmissionHandler

from .fakes import InMemoryIntentRepository, StoreErrorCategory


@pytest.fixture
def repository(
    scope: SubscriptionScope, intent_document: IntentDocument
) -> InMemoryIntentRepository:
    repository = InMemoryIntentRepository(scope)
    repository.initialize(intent_document)
    return repository


@pytest.fixture
def handler(
    scope: SubscriptionScope, repository: InMemoryIntentRepository
) -> AdmissionHandler:
    return DrasiAdmissionHandler(scope=scope, intents=repository)


def _save_intent(
    repository: InMemoryIntentRepository, intent: SubscriptionIntent
) -> None:
    snapshot = repository.load()
    assert snapshot is not None
    snapshot.document.intents[intent.query_id] = intent
    repository.save(snapshot.document, expected_etag=snapshot.etag)


@pytest.mark.parametrize("field", ("router_id", "namespace", "app_id", "agent_name"))
def test_constructor_rejects_a_differently_scoped_reader(
    scope: SubscriptionScope,
    repository: InMemoryIntentRepository,
    field: str,
) -> None:
    document = scope.model_dump()
    document[field] = "other/router" if field == "router_id" else "other"
    other_scope = SubscriptionScope.model_validate(document)

    with pytest.raises(ValueError, match="scopes must match"):
        DrasiAdmissionHandler(scope=other_scope, intents=repository)


def test_active_allowed_events_produce_tasks_without_changing_state(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    active_intent: SubscriptionIntent,
    canonical_delivery: AgentDelivery,
    mocker: MockerFixture,
) -> None:
    active_intent.operations = tuple(Operation)
    _save_intent(repository, active_intent)
    original = repository.load()
    get = mocker.spy(repository, "get")
    document = to_wire(canonical_delivery)

    result = handler.admit(document)

    assert isinstance(result, SchedulingInput)
    assert result.event_id == canonical_delivery.eventId
    assert active_intent.instructions in result.task
    assert repository.load() == original
    assert document == to_wire(canonical_delivery)
    get.assert_called_once_with(active_intent.query_id)


@pytest.mark.parametrize("initialized", (False, True))
def test_absent_document_or_query_is_an_intentional_discard(
    scope: SubscriptionScope, insert_delivery: AgentDelivery, initialized: bool
) -> None:
    repository = InMemoryIntentRepository(scope)
    if initialized:
        repository.initialize(IntentDocument(format_version=1, scope=scope, intents={}))
    handler = DrasiAdmissionHandler(scope=scope, intents=repository)

    assert handler.admit(to_wire(insert_delivery)) == Discard(reason="no_intent")


@pytest.mark.parametrize("included", (False, True))
@pytest.mark.parametrize(
    ("status", "expected"),
    (
        ("pending_subscribe", Retry(reason="pending_subscription")),
        ("pending_update", Retry(reason="pending_subscription")),
        ("pending_unsubscribe", Discard(reason="pending_unsubscribe")),
        ("unavailable", Discard(reason="unavailable")),
    ),
)
def test_lifecycle_is_resolved_before_operation_filtering(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    status_intents: dict[SubscriptionStatus, SubscriptionIntent],
    insert_delivery: AgentDelivery,
    included: bool,
    status: SubscriptionStatus,
    expected: Discard | Retry,
) -> None:
    intent = status_intents[status]
    intent.operations = (Operation.i,) if included else (Operation.u,)
    _save_intent(repository, intent)

    assert handler.admit(to_wire(insert_delivery)) == expected


def test_active_excluded_operations_are_discarded(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    active_intent: SubscriptionIntent,
    canonical_delivery: AgentDelivery,
) -> None:
    active_intent.operations = (
        Operation.u if canonical_delivery.event.op != "u" else Operation.i,
    )
    _save_intent(repository, active_intent)

    assert handler.admit(to_wire(canonical_delivery)) == Discard(
        reason="operation_excluded"
    )


@pytest.mark.parametrize(
    "status",
    (
        "pending_subscribe",
        "pending_update",
        "active",
        "pending_unsubscribe",
        "unavailable",
    ),
)
def test_stale_incarnations_never_schedule_or_wait_for_pending_transitions(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    status_intents: dict[SubscriptionStatus, SubscriptionIntent],
    canonical_delivery: AgentDelivery,
    status: SubscriptionStatus,
) -> None:
    _save_intent(repository, status_intents[status])
    document = to_wire(canonical_delivery)
    document["subscriptionIncarnation"] = "retired-incarnation"

    assert handler.admit(document) == Discard(reason="stale_incarnation")


@pytest.mark.parametrize(
    ("category", "expected"),
    (
        ("unavailable", Retry(reason="state_unavailable")),
        ("corrupt", Retry(reason="state_corrupt")),
        ("unsupported_version", Retry(reason="unsupported_state_version")),
        ("conflict", Retry(reason="state_unavailable")),
    ),
)
def test_store_errors_retry_instead_of_becoming_absent_intent(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    insert_delivery: AgentDelivery,
    category: StoreErrorCategory,
    expected: Retry,
) -> None:
    repository.fail_next("get", category)

    assert handler.admit(to_wire(insert_delivery)) == expected


def test_shared_invalid_deliveries_poison_before_reading_intent(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    insert_delivery: AgentDelivery,
) -> None:
    resource = files("drasi_agent_router_contracts").joinpath(
        "fixtures", "messages.json"
    )
    cases: list[dict[str, Any]] = json.loads(resource.read_text(encoding="utf-8"))
    invalid = [
        case for case in cases if case["model"] == "AgentDelivery" and not case["valid"]
    ]
    assert invalid
    repository.fail_next("get", "unavailable")

    for case in invalid:
        assert handler.admit(case["message"]) == Poison(reason="invalid_delivery"), (
            case["name"]
        )
    assert handler.admit(to_wire(insert_delivery)) == Retry(reason="state_unavailable")


@pytest.mark.parametrize("data", (None, [], True, 42, "{}", b"{}"))
def test_non_object_inputs_are_poison(handler: AdmissionHandler, data: object) -> None:
    assert handler.admit(data) == Poison(reason="invalid_delivery")


@pytest.mark.parametrize("shape", ("static", "cloudevent", "encoded_json"))
def test_only_decoded_m2_data_is_accepted(
    handler: AdmissionHandler, insert_delivery: AgentDelivery, shape: str
) -> None:
    document = to_wire(insert_delivery)
    data: object
    if shape == "static":
        data = document["event"]
    elif shape == "cloudevent":
        data = {"specversion": "1.0", "id": "publication-id", "data": document}
    else:
        data = json.dumps(document)

    assert handler.admit(data) == Poison(reason="invalid_delivery")


@pytest.mark.parametrize(
    "value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
        (1, 2),
        {"nested": [float("nan")]},
        {1: "non-string key"},
        b"bytes",
        object(),
    ),
)
def test_non_json_row_values_poison_before_serialization_or_state_access(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    insert_delivery: AgentDelivery,
    value: object,
) -> None:
    repository.fail_next("get", "unavailable")
    document = to_wire(insert_delivery)
    document["event"]["payload"]["after"]["invalid"] = value

    assert handler.admit(document) == Poison(reason="invalid_delivery")
    assert handler.admit(to_wire(insert_delivery)) == Retry(reason="state_unavailable")


@pytest.mark.parametrize("overflow", (False, True))
def test_event_integer_encoding_limit_has_a_delivery_disposition(
    handler: AdmissionHandler,
    insert_delivery: AgentDelivery,
    overflow: bool,
) -> None:
    limit = sys.get_int_max_str_digits()
    if limit == 0:
        pytest.skip("The interpreter's integer-string conversion limit is disabled.")
    value = 10 ** (limit - 1 + int(overflow))
    document = to_wire(insert_delivery)
    document["event"]["payload"]["after"]["large_integer"] = value
    parse(AgentDelivery, document)

    result = handler.admit(document)

    if overflow:
        assert result == Poison(reason="invalid_delivery")
    else:
        assert isinstance(result, SchedulingInput)
        event = json.loads(
            result.task.split("BEGIN_UNTRUSTED_DRASI_EVENT_JSON\n", 1)[1].split(
                "\n", 1
            )[0]
        )
        assert event["payload"]["after"]["large_integer"] == value
    assert sys.get_int_max_str_digits() == limit


def test_unrenderable_event_is_poison_before_reading_intent_without_payload_logs(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    insert_delivery: AgentDelivery,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    limit = sys.get_int_max_str_digits()
    if limit == 0:
        pytest.skip("The interpreter's integer-string conversion limit is disabled.")
    document = to_wire(insert_delivery)
    document["event"]["payload"]["after"]["ROW_SENTINEL"] = 10**limit
    repository.fail_next("get", "unavailable")
    get = mocker.spy(repository, "get")

    with caplog.at_level(logging.DEBUG):
        assert handler.admit(document) == Poison(reason="invalid_delivery")

    get.assert_not_called()
    assert "invalid_delivery" in caplog.text
    assert "SENTINEL" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert handler.admit(to_wire(insert_delivery)) == Retry(reason="state_unavailable")


def test_wrong_router_is_poison_before_state_access(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    insert_delivery: AgentDelivery,
) -> None:
    repository.fail_next("get", "unavailable")
    document = to_wire(insert_delivery)
    document["routerId"] = "other/router"

    assert handler.admit(document) == Poison(reason="router_mismatch")
    assert handler.admit(to_wire(insert_delivery)) == Retry(reason="state_unavailable")


@pytest.mark.parametrize("corruption", ("query", "router", "status", "catalog"))
def test_inconsistent_reader_snapshots_are_not_scheduled(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    active_intent: SubscriptionIntent,
    insert_delivery: AgentDelivery,
    mocker: MockerFixture,
    corruption: str,
) -> None:
    if corruption == "query":
        active_intent.query_id = "rollout-status"
    elif corruption == "router":
        active_intent.catalog_snapshot.router_id = "other/router"
    elif corruption == "status":
        active_intent = active_intent.model_copy(update={"status": "unknown"})
    else:
        active_intent.catalog_snapshot.queries[1].usage = None
    mocker.patch.object(repository, "get", return_value=active_intent)

    assert handler.admit(to_wire(insert_delivery)) == Retry(reason="state_corrupt")


def test_each_admission_reads_latest_instructions_without_rewriting_prior_work(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    insert_delivery: AgentDelivery,
) -> None:
    document = to_wire(insert_delivery)
    first = handler.admit(document)
    assert isinstance(first, SchedulingInput)
    original_task = first.task
    snapshot = repository.load()
    assert snapshot is not None
    intent = snapshot.document.intents["service-errors"]
    intent.instructions = "Use the updated error-handling procedure."
    intent.catalog_snapshot.queries[0].title = "Updated catalog context"
    repository.save(snapshot.document, expected_etag=snapshot.etag)

    second = handler.admit(document)

    assert isinstance(second, SchedulingInput)
    assert second.instance_id == first.instance_id
    assert second.task != first.task
    assert intent.instructions in second.task
    assert "Updated catalog context" in second.task
    assert first.task == original_task
    assert intent.instructions not in first.task


def test_unexpected_reader_errors_are_not_reclassified_as_success(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    insert_delivery: AgentDelivery,
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(repository, "get", side_effect=RuntimeError("reader bug"))

    with pytest.raises(RuntimeError, match="reader bug"):
        handler.admit(to_wire(insert_delivery))


def test_failure_logs_do_not_contain_rows_instructions_or_validation_tracebacks(
    handler: AdmissionHandler,
    repository: InMemoryIntentRepository,
    active_intent: SubscriptionIntent,
    insert_delivery: AgentDelivery,
    caplog: pytest.LogCaptureFixture,
) -> None:
    active_intent.instructions = "INSTRUCTION_SENTINEL"
    _save_intent(repository, active_intent)
    document = to_wire(insert_delivery)
    document["event"]["payload"]["after"]["private"] = "ROW_SENTINEL"

    with caplog.at_level(logging.DEBUG):
        document["event"]["seq"] = "INVALID_SEQUENCE_SENTINEL"
        assert handler.admit(document) == Poison(reason="invalid_delivery")
        document["event"]["seq"] = insert_delivery.event.seq
        document["routerId"] = "other/router"
        assert handler.admit(document) == Poison(reason="router_mismatch")
        document["routerId"] = insert_delivery.routerId
        repository.fail_next("get", "unavailable")
        assert handler.admit(document) == Retry(reason="state_unavailable")

    assert len(caplog.records) == 3
    assert "SENTINEL" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
