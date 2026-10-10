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

"""Exercise the real state wrappers with a persistent, injected SDK boundary."""

# mypy: ignore-errors=False

from __future__ import annotations

import json
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from unittest.mock import MagicMock

import pytest
from dapr.clients import DaprClient
from dapr.clients.exceptions import DaprGrpcError, DaprInternalError
from dapr.clients.grpc._response import StateResponse
from dapr.clients.grpc._state import Concurrency, Consistency, StateOptions
from dapr.clients.retry import RetryPolicy
from dapr.proto import api_v1
from grpc import RpcError, StatusCode
from pydantic import BaseModel

from dapr_agents.agents.configs import AgentStateConfig
from dapr_agents.ext.drasi._interfaces import (
    IntentReader,
    IntentRepository,
    IntentStoreError,
)
from dapr_agents.ext.drasi._models import (
    IntentDocument,
    RouterOperationOutcome,
    SubscriptionIntent,
    SubscriptionScope,
    SubscriptionStatus,
)
from dapr_agents.ext.drasi.intent_store import DaprIntentRepository
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.storage.daprstores.statestore import DaprStateStore

_PRIVATE_DETAIL = "private-instructions-and-backend-credentials"
_STORE_NAME = "custom-runtime"
_PREFIX = "tenant:"


class _RpcFailure(RpcError):
    def __init__(self, status: StatusCode) -> None:
        self._status = status

    def code(self) -> StatusCode:
        return self._status

    def details(self) -> str:
        return _PRIVATE_DETAIL

    def trailing_metadata(self) -> tuple[()]:
        return ()


def _grpc_error(status: StatusCode) -> DaprGrpcError:
    return DaprGrpcError(_RpcFailure(status))


def _native_sdk_client(stub: MagicMock, retry_policy: RetryPolicy) -> DaprClient:
    """Keep real SDK methods and retries; replace only the gRPC transport."""
    client = object.__new__(DaprClient)
    client._stub = stub
    client._channel = MagicMock()
    client.retry_policy = retry_policy
    return client


@dataclass(frozen=True)
class _Write:
    store_name: str
    key: str
    value: str | bytes
    etag: str | None
    options: StateOptions | None
    metadata: dict[str, str] | None


class _Backend:
    """Keep bytes and ETags across newly constructed services and SDK clients."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], StateResponse] = {}
        self.reads: list[tuple[str, str]] = []
        self.writes: list[_Write] = []
        self.clients: list[MagicMock] = []
        self.read_error: Exception | None = None
        self.write_error: Exception | None = None
        self.after_write_error: Exception | None = None
        self._revision = 0
        self._lock = Lock()

    def client(self) -> DaprClient:
        client = MagicMock(spec=DaprClient)
        client.__enter__.return_value = client
        client.get_state.side_effect = self.get_state
        client.save_state.side_effect = self.save_state
        client.retry_policy = RetryPolicy(max_attempts=0)
        client._stub = MagicMock()
        client._stub.GetState.with_call.side_effect = self.get_rpc
        self.clients.append(client)
        return client

    def get_rpc(
        self, request: api_v1.GetStateRequest, *, metadata: object
    ) -> tuple[StateResponse, MagicMock]:
        assert request.consistency == Consistency.strong.value
        return (
            self.get_state(
                store_name=request.store_name,
                key=request.key,
                state_metadata=dict(request.metadata),
            ),
            MagicMock(),
        )

    def get_state(
        self, *, store_name: str, key: str, state_metadata: dict[str, str] | None
    ) -> StateResponse:
        with self._lock:
            self.reads.append((store_name, key))
            if self.read_error is not None:
                raise self.read_error
            record = self.records.get((store_name, key), StateResponse(b""))
            return StateResponse(record.data, record.etag)

    def save_state(
        self,
        *,
        store_name: str,
        key: str,
        value: str | bytes,
        state_metadata: dict[str, str] | None,
        etag: str | None,
        options: StateOptions | None,
    ) -> None:
        with self._lock:
            self.writes.append(
                _Write(store_name, key, value, etag, options, state_metadata)
            )
            if self.write_error is not None:
                raise self.write_error
            existing = self.records.get((store_name, key))
            if etag is not None and (existing is None or existing.etag != etag):
                raise _grpc_error(StatusCode.ABORTED)
            self._revision += 1
            self.records[(store_name, key)] = StateResponse(
                value, f"revision-{self._revision}"
            )
            if self.after_write_error is not None:
                raise self.after_write_error


@pytest.fixture
def backend() -> _Backend:
    return _Backend()


def _state_config(backend: _Backend) -> AgentStateConfig:
    return AgentStateConfig(
        store=StateStoreService(
            store_name=_STORE_NAME,
            key_prefix=_PREFIX,
            client_factory=backend.client,
        ),
        state_key_prefix="workflow-entries-only",
    )


@pytest.fixture
def repository(scope: SubscriptionScope, backend: _Backend) -> DaprIntentRepository:
    return DaprIntentRepository(scope=scope, store=_state_config(backend).store)


def _key(scope: SubscriptionScope) -> tuple[str, str]:
    return _STORE_NAME, f"{_PREFIX}drasi:intent:{scope.inbox_topic}"


def test_absent_document_and_query_do_not_create_state(
    repository: DaprIntentRepository,
    backend: _Backend,
    intent_document: IntentDocument,
) -> None:
    reader: IntentReader = repository
    writable: IntentRepository = repository
    assert writable.load() is None
    assert reader.get("service-errors") is None
    assert backend.writes == []

    repository.initialize(intent_document)
    assert repository.get("not-in-the-catalog") is None
    assert len(backend.writes) == 1


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
def test_intent_survives_service_client_and_repository_reconstruction(
    repository: DaprIntentRepository,
    backend: _Backend,
    scope: SubscriptionScope,
    status_intents: dict[SubscriptionStatus, SubscriptionIntent],
    status: SubscriptionStatus,
) -> None:
    intent = status_intents[status]
    intent.last_outcome = RouterOperationOutcome(
        operation="unsubscribe", outcome="uncertain", error_category="transport"
    )
    document = IntentDocument(
        format_version=1, scope=scope, intents={intent.query_id: intent}
    )
    repository.initialize(document)

    restored = DaprIntentRepository(scope=scope, store=_state_config(backend).store)
    snapshot = restored.load()
    assert snapshot is not None
    assert snapshot.document == document
    assert snapshot.etag
    assert restored.get(intent.query_id) == intent
    assert restored.scope == scope
    assert len(backend.clients) == 3
    for client in backend.clients:
        client.__exit__.assert_called_once()

    wire = json.loads(backend.records[_key(scope)].data)
    assert (
        "usage"
        not in wire["intents"][intent.query_id]["catalog_snapshot"]["queries"][1]
    )


def test_configured_primitive_prefix_and_options_are_preserved(
    backend: _Backend,
    scope: SubscriptionScope,
    intent_document: IntentDocument,
    tmp_path: Path,
) -> None:
    class WorkflowOnly(BaseModel):
        workflow_instance_id: str

    primitive = DaprStateStore(store_name=_STORE_NAME, client_factory=backend.client)
    factory = MagicMock(return_value=primitive)
    unused_client_factory = MagicMock(side_effect=AssertionError("wrong factory"))
    service = StateStoreService(
        store_name=_STORE_NAME,
        key_prefix=_PREFIX,
        model=WorkflowOnly,
        store_factory=factory,
        client_factory=unused_client_factory,
        retry_attempts=7,
        mirror_writes=True,
        local_mirror_path=str(tmp_path),
    )
    repository = DaprIntentRepository(scope=scope, store=service)
    repository.initialize(intent_document)
    snapshot = repository.load()
    assert snapshot is not None
    repository.save(snapshot.document, expected_etag=snapshot.etag)

    factory.assert_called_once_with()
    unused_client_factory.assert_not_called()
    assert service.model is WorkflowOnly
    assert service.retry_attempts == 7
    assert service.mirror_writes is True
    assert list(tmp_path.iterdir()) == []
    assert set(backend.records) == {_key(scope)}
    assert backend.writes[0].etag is None
    assert backend.writes[1].etag == snapshot.etag
    for write in backend.writes:
        assert write.options is not None
        assert write.options.concurrency == Concurrency.first_write
        assert write.options.consistency == Consistency.strong
        assert write.metadata == {
            "contentType": "application/json",
            "partitionKey": _key(scope)[1],
        }
    request = backend.clients[1]._stub.GetState.with_call.call_args.args[0]
    assert request.store_name == _STORE_NAME
    assert request.key == _key(scope)[1]
    assert dict(request.metadata) == backend.writes[0].metadata
    assert request.consistency == Consistency.strong.value


def test_initialize_is_unconditional_and_etags_require_reloading(
    repository: DaprIntentRepository,
    intent_document: IntentDocument,
) -> None:
    repository.initialize(intent_document)
    first = repository.load()
    assert first is not None
    intent_document.intents["service-errors"].instructions = "Replacement"
    repository.initialize(intent_document)
    second = repository.load()
    assert second is not None
    assert second.etag != first.etag
    assert second.document == intent_document
    repository.save(second.document, expected_etag=second.etag)
    saved = repository.load()
    assert saved is not None
    assert saved.etag != second.etag


def test_different_query_writers_conflict_and_require_a_fresh_merge(
    repository: DaprIntentRepository,
    backend: _Backend,
    scope: SubscriptionScope,
    intent_document: IntentDocument,
) -> None:
    repository.initialize(intent_document)
    other = DaprIntentRepository(scope=scope, store=_state_config(backend).store)
    first, second = repository.load(), other.load()
    assert first is not None and second is not None
    assert first.etag == second.etag
    first.document.intents["service-errors"].instructions = "First writer"
    repository.save(first.document, expected_etag=first.etag)
    second.document.intents["rollout-status"].instructions = "Second writer"

    with pytest.raises(IntentStoreError) as failure:
        other.save(second.document, expected_etag=second.etag)
    assert failure.value.category == "conflict"
    assert len(backend.writes) == 3
    fresh = other.load()
    assert fresh is not None
    assert fresh.document.intents["service-errors"].instructions == "First writer"
    assert (
        fresh.document.intents["rollout-status"]
        == intent_document.intents["rollout-status"]
    )
    fresh.document.intents["rollout-status"].instructions = "Second writer"
    other.save(fresh.document, expected_etag=fresh.etag)
    final = repository.load()
    assert final is not None
    assert final.document.intents["service-errors"].instructions == "First writer"
    assert final.document.intents["rollout-status"].instructions == "Second writer"


def test_save_never_initializes_missing_state_or_accepts_an_empty_etag(
    repository: DaprIntentRepository,
    backend: _Backend,
    intent_document: IntentDocument,
) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        repository.save(intent_document, expected_etag="")
    assert backend.writes == []
    with pytest.raises(IntentStoreError) as failure:
        repository.save(intent_document, expected_etag="missing-document")
    assert failure.value.category == "conflict"
    assert backend.records == {}
    assert len(backend.writes) == 1


def test_nested_inputs_and_return_values_are_detached(
    repository: DaprIntentRepository, intent_document: IntentDocument
) -> None:
    original = intent_document.model_dump_json()
    repository.initialize(intent_document)
    intent_document.intents["service-errors"].instructions = "Caller mutation"
    intent_document.intents["service-errors"].catalog_snapshot.queries.clear()
    snapshot = repository.load()
    assert snapshot is not None
    assert snapshot.document.model_dump_json() == original
    snapshot.document.intents.clear()
    fetched = repository.get("service-errors")
    assert fetched is not None
    fetched.catalog_snapshot.queries.clear()

    update = repository.load()
    assert update is not None
    update.document.intents["service-errors"].instructions = "Saved update"
    repository.save(update.document, expected_etag=update.etag)
    update.document.intents["service-errors"].instructions = "Post-save mutation"
    reread = repository.get("service-errors")
    assert reread is not None
    assert reread.instructions == "Saved update"
    assert reread.catalog_snapshot.queries


def test_empty_document_is_retained_after_conditional_removal(
    repository: DaprIntentRepository,
    backend: _Backend,
    intent_document: IntentDocument,
) -> None:
    repository.initialize(intent_document)
    snapshot = repository.load()
    assert snapshot is not None
    snapshot.document.intents.clear()
    repository.save(snapshot.document, expected_etag=snapshot.etag)
    empty = repository.load()
    assert empty is not None
    assert empty.document.intents == {}
    assert empty.etag != snapshot.etag
    assert len(backend.records) == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("router_id", "another/router"),
        ("namespace", "another-namespace"),
        ("app_id", "another-app"),
        ("agent_name", "checkoutsre"),
        ("agent_name", "Checkout SRE"),
    ),
)
def test_each_exact_identity_is_isolated(
    repository: DaprIntentRepository,
    backend: _Backend,
    scope: SubscriptionScope,
    intent_document: IntentDocument,
    field: str,
    replacement: str,
) -> None:
    repository.initialize(intent_document)
    identity = scope.model_dump()
    identity[field] = replacement
    other_scope = SubscriptionScope.model_validate(identity)
    other = DaprIntentRepository(scope=other_scope, store=_state_config(backend).store)
    assert other.load() is None
    assert other.get("service-errors") is None
    other.initialize(IntentDocument(format_version=1, scope=other_scope, intents={}))
    assert len(backend.records) == 2
    assert repository.get("service-errors") == intent_document.intents["service-errors"]

    with pytest.raises(ValueError, match="scope"):
        other.initialize(intent_document)
    backend.records[_key(other_scope)] = backend.records[_key(scope)]
    with pytest.raises(IntentStoreError) as failure:
        other.get("not-in-the-document")
    assert failure.value.category == "corrupt"


@pytest.mark.parametrize(
    "data",
    (
        b"",
        b"{",
        b"\xff",
        b"null",
        b"[]",
        b"{}",
        b"42",
        pytest.param(b"9" * 5000, id="oversized-integer"),
        pytest.param(b"[" * 1500 + b"0" + b"]" * 1500, id="deeply-nested"),
    ),
)
def test_corrupt_state_is_not_absence(
    repository: DaprIntentRepository,
    backend: _Backend,
    scope: SubscriptionScope,
    data: bytes,
) -> None:
    backend.records[_key(scope)] = StateResponse(data, "existing")
    with pytest.raises(IntentStoreError) as failure:
        repository.get("service-errors")
    assert failure.value.category == "corrupt"
    assert backend.writes == []


@pytest.mark.parametrize(
    ("version", "category"),
    (
        (2, "unsupported_version"),
        (0, "unsupported_version"),
        (True, "corrupt"),
        ("1", "corrupt"),
        (1.0, "corrupt"),
        (None, "corrupt"),
    ),
)
def test_version_failures_do_not_reset_state(
    repository: DaprIntentRepository,
    backend: _Backend,
    scope: SubscriptionScope,
    intent_document: IntentDocument,
    version: object,
    category: str,
) -> None:
    wire = intent_document.model_dump(mode="json")
    wire["format_version"] = version
    data = json.dumps(wire)
    backend.records[_key(scope)] = StateResponse(data, "existing")
    with pytest.raises(IntentStoreError) as failure:
        repository.load()
    assert failure.value.category == category
    assert backend.records[_key(scope)].text() == data
    assert backend.writes == []


def test_existing_document_without_etag_is_unavailable(
    repository: DaprIntentRepository,
    backend: _Backend,
    scope: SubscriptionScope,
    intent_document: IntentDocument,
) -> None:
    backend.records[_key(scope)] = StateResponse(intent_document.model_dump_json())
    with pytest.raises(IntentStoreError) as failure:
        repository.load()
    assert failure.value.category == "unavailable"


@pytest.mark.parametrize(
    "error",
    (
        _grpc_error(StatusCode.UNAVAILABLE),
        _grpc_error(StatusCode.DEADLINE_EXCEEDED),
        DaprInternalError(_PRIVATE_DETAIL),
        TimeoutError(_PRIVATE_DETAIL),
        ConnectionError(_PRIVATE_DETAIL),
    ),
)
def test_read_failures_are_safe_and_never_empty(
    repository: DaprIntentRepository,
    backend: _Backend,
    error: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend.read_error = error
    with pytest.raises(IntentStoreError) as failure:
        repository.get("service-errors")
    assert failure.value.category == "unavailable"
    assert len(backend.reads) == 1
    assert backend.writes == []
    assert _PRIVATE_DETAIL not in caplog.text
    assert _PRIVATE_DETAIL not in "".join(traceback.format_exception(failure.value))


@pytest.mark.parametrize(
    ("status", "category"),
    (
        (StatusCode.ABORTED, "conflict"),
        (StatusCode.INTERNAL, "unavailable"),
        (StatusCode.INVALID_ARGUMENT, "unavailable"),
        (StatusCode.UNAVAILABLE, "unavailable"),
        (StatusCode.DEADLINE_EXCEEDED, "unavailable"),
    ),
)
def test_save_failures_are_classified_without_retry_or_payload_leakage(
    repository: DaprIntentRepository,
    backend: _Backend,
    intent_document: IntentDocument,
    status: StatusCode,
    category: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    repository.initialize(intent_document)
    snapshot = repository.load()
    assert snapshot is not None
    backend.write_error = _grpc_error(status)
    with pytest.raises(IntentStoreError) as failure:
        repository.save(snapshot.document, expected_etag=snapshot.etag)
    assert failure.value.category == category
    assert len(backend.writes) == 2
    assert _PRIVATE_DETAIL not in caplog.text
    assert _PRIVATE_DETAIL not in "".join(traceback.format_exception(failure.value))


@pytest.mark.parametrize(
    "error",
    (
        _RpcFailure(StatusCode.INTERNAL),
        DaprInternalError(_PRIVATE_DETAIL),
        TimeoutError(_PRIVATE_DETAIL),
        ConnectionError(_PRIVATE_DETAIL),
    ),
)
def test_initialization_transport_failures_are_not_success(
    repository: DaprIntentRepository,
    backend: _Backend,
    intent_document: IntentDocument,
    error: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend.write_error = error
    with pytest.raises(IntentStoreError) as failure:
        repository.initialize(intent_document)
    assert failure.value.category == "unavailable"
    assert backend.records == {}
    assert len(backend.writes) == 1
    assert _PRIVATE_DETAIL not in caplog.text
    assert _PRIVATE_DETAIL not in "".join(traceback.format_exception(failure.value))


@pytest.mark.parametrize("operation", ("initialize", "save"))
def test_ambiguous_write_is_recovered_by_reloading(
    repository: DaprIntentRepository,
    backend: _Backend,
    intent_document: IntentDocument,
    operation: str,
) -> None:
    repository.initialize(intent_document)
    snapshot = repository.load()
    assert snapshot is not None
    snapshot.document.intents["service-errors"].status = "pending_unsubscribe"
    backend.after_write_error = _grpc_error(StatusCode.DEADLINE_EXCEEDED)
    with pytest.raises(IntentStoreError) as failure:
        if operation == "initialize":
            repository.initialize(snapshot.document)
        else:
            repository.save(snapshot.document, expected_etag=snapshot.etag)
    assert failure.value.category == "unavailable"
    assert len(backend.writes) == 2
    restored = repository.load()
    assert restored is not None
    assert restored.etag != snapshot.etag
    assert restored.document == snapshot.document


def test_invalid_nested_input_never_reaches_storage_or_logs_payload(
    repository: DaprIntentRepository,
    backend: _Backend,
    scope: SubscriptionScope,
    intent_document: IntentDocument,
    caplog: pytest.LogCaptureFixture,
) -> None:
    wire = intent_document.model_dump(mode="json")
    wire["intents"]["service-errors"]["instructions"] = {"private": _PRIVATE_DETAIL}
    backend.records[_key(scope)] = StateResponse(json.dumps(wire), "existing")
    with pytest.raises(IntentStoreError) as failure:
        repository.load()
    assert failure.value.category == "corrupt"
    assert _PRIVATE_DETAIL not in "".join(traceback.format_exception(failure.value))

    intent_document.intents["wrong-query-key"] = intent_document.intents[
        "service-errors"
    ]
    with pytest.raises(IntentStoreError) as failure:
        repository.initialize(intent_document)
    assert failure.value.category == "corrupt"
    assert backend.writes == []
    assert _PRIVATE_DETAIL not in caplog.text


@pytest.mark.parametrize("field", ("instructions", "catalog"))
def test_invalid_model_serialization_does_not_emit_sensitive_warnings(
    repository: DaprIntentRepository,
    backend: _Backend,
    intent_document: IntentDocument,
    caplog: pytest.LogCaptureFixture,
    field: str,
) -> None:
    intent = intent_document.intents["service-errors"]
    if field == "instructions":
        intent_document.intents["service-errors"] = intent.model_copy(
            update={"instructions": {"private": _PRIVATE_DETAIL}}
        )
    else:
        query = intent.catalog_snapshot.queries[0]
        intent.catalog_snapshot.queries[0] = query.model_copy(
            update={"title": {"private": _PRIVATE_DETAIL}}
        )
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(IntentStoreError) as failure:
            repository.initialize(intent_document)
    assert failure.value.category == "corrupt"
    assert backend.writes == []
    assert _PRIVATE_DETAIL not in caplog.text
    assert _PRIVATE_DETAIL not in "".join(traceback.format_exception(failure.value))
    assert all(_PRIVATE_DETAIL not in str(warning.message) for warning in emitted)


def test_programming_errors_are_not_disguised_as_storage_failures(
    repository: DaprIntentRepository, backend: _Backend
) -> None:
    backend.read_error = TypeError("broken client implementation")
    with pytest.raises(TypeError, match="broken client"):
        repository.load()


@pytest.mark.parametrize("operation", ("initialize", "save"))
@pytest.mark.parametrize(
    "status", (StatusCode.UNAVAILABLE, StatusCode.DEADLINE_EXCEEDED)
)
def test_uncertain_writes_do_not_retry_inside_the_native_sdk(
    scope: SubscriptionScope,
    intent_document: IntentDocument,
    operation: str,
    status: StatusCode,
) -> None:
    persisted = (
        api_v1.GetStateResponse(
            data=intent_document.model_dump_json().encode(), etag="before"
        )
        if operation == "save"
        else api_v1.GetStateResponse()
    )
    stub = MagicMock()
    rpc_call = MagicMock()
    rpc_call.initial_metadata.return_value = ()

    def get_rpc(
        request: api_v1.GetStateRequest, *, metadata: object
    ) -> tuple[api_v1.GetStateResponse, MagicMock]:
        return persisted, rpc_call

    def save_rpc(request: api_v1.SaveStateRequest, *, metadata: object) -> None:
        nonlocal persisted
        item = request.states[0]
        if item.HasField("etag") and item.etag.value != persisted.etag:
            raise _RpcFailure(StatusCode.ABORTED)
        persisted = api_v1.GetStateResponse(data=item.value, etag="committed")
        raise _RpcFailure(status)

    stub.GetState.with_call.side_effect = get_rpc
    stub.SaveState.with_call.side_effect = save_rpc
    configured_policy = RetryPolicy(max_attempts=2)
    clients: list[DaprClient] = []

    def factory() -> DaprClient:
        client = _native_sdk_client(stub, configured_policy)
        clients.append(client)
        return client

    repository = DaprIntentRepository(
        scope=scope,
        store=StateStoreService(
            store_name=_STORE_NAME, key_prefix=_PREFIX, client_factory=factory
        ),
    )
    intent_document.intents["service-errors"].status = "pending_unsubscribe"
    with pytest.raises(IntentStoreError) as failure:
        if operation == "save":
            repository.save(intent_document, expected_etag="before")
        else:
            repository.initialize(intent_document)
    assert failure.value.category == "unavailable"
    assert stub.SaveState.with_call.call_count == 1
    assert clients[0].retry_policy.max_attempts == 0
    assert configured_policy.max_attempts == 2

    restored = repository.load()
    assert restored is not None
    assert restored.document == intent_document
    assert restored.etag == "committed"
    assert clients[1].retry_policy is configured_policy
    request = stub.SaveState.with_call.call_args.args[0]
    assert request.store_name == _STORE_NAME
    assert request.states[0].key == _key(scope)[1]
    assert request.states[0].HasField("etag") == (operation == "save")
    assert request.states[0].options.concurrency == Concurrency.first_write.value
    assert request.states[0].options.consistency == Consistency.strong.value


@pytest.mark.parametrize(
    ("empty_payload", "etag", "error_category"),
    (
        (True, "", None),
        (True, "0", "corrupt"),
        (False, "", "unavailable"),
        (False, "0", None),
    ),
)
def test_native_sdk_missing_state_and_string_etag_semantics(
    scope: SubscriptionScope,
    intent_document: IntentDocument,
    empty_payload: bool,
    etag: str,
    error_category: str | None,
) -> None:
    response = api_v1.GetStateResponse(
        data=b"" if empty_payload else intent_document.model_dump_json().encode(),
        etag=etag,
    )
    stub = MagicMock()
    rpc_call = MagicMock()
    rpc_call.initial_metadata.return_value = ()
    stub.GetState.with_call.return_value = response, rpc_call
    repository = DaprIntentRepository(
        scope=scope,
        store=StateStoreService(
            store_name=_STORE_NAME,
            client_factory=lambda: _native_sdk_client(
                stub, RetryPolicy(max_attempts=0)
            ),
        ),
    )

    if error_category is not None:
        with pytest.raises(IntentStoreError) as failure:
            repository.load()
        assert failure.value.category == error_category
    else:
        snapshot = repository.load()
        if empty_payload:
            assert response.etag == ""
            assert snapshot is None
        else:
            assert snapshot is not None
            assert snapshot.document == intent_document
            assert snapshot.etag == "0"
    stub.SaveState.with_call.assert_not_called()


@pytest.mark.parametrize("pending_unsubscribe", (False, True))
def test_repository_uses_strong_reads_instead_of_stale_replica_intent(
    scope: SubscriptionScope,
    active_intent: SubscriptionIntent,
    pending_unsubscribe: bool,
) -> None:
    active = IntentDocument(
        format_version=1, scope=scope, intents={active_intent.query_id: active_intent}
    )
    if pending_unsubscribe:
        stale = active
        current = active.model_copy(deep=True)
        current.intents[active_intent.query_id].status = "pending_unsubscribe"
    else:
        stale = IntentDocument(format_version=1, scope=scope, intents={})
        current = active

    stub = MagicMock()

    def get_rpc(
        request: api_v1.GetStateRequest, *, metadata: object
    ) -> tuple[api_v1.GetStateResponse, MagicMock]:
        document = current if request.consistency == Consistency.strong.value else stale
        return (
            api_v1.GetStateResponse(
                data=document.model_dump_json().encode(), etag="current"
            ),
            MagicMock(),
        )

    stub.GetState.with_call.side_effect = get_rpc
    repository = DaprIntentRepository(
        scope=scope,
        store=StateStoreService(
            store_name=_STORE_NAME,
            client_factory=lambda: _native_sdk_client(
                stub, RetryPolicy(max_attempts=0)
            ),
        ),
    )
    assert (
        repository.get(active_intent.query_id)
        == current.intents[active_intent.query_id]
    )
    request = stub.GetState.with_call.call_args.args[0]
    assert request.consistency == Consistency.strong.value
    stub.SaveState.with_call.assert_not_called()
