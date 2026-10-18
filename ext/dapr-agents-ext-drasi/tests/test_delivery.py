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

"""Dapr inbox dispositions, durable scheduling acceptance, and stream lifetime."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from functools import partial
from queue import Queue
from threading import Event, Thread
from unittest.mock import MagicMock, Mock

import pytest
from dapr.clients import DaprClient
from dapr.clients.grpc._response import TopicEventResponse, TopicEventResponseStatus
from dapr.clients.grpc.subscription import Subscription
from dapr.common.pubsub.subscription import (
    StreamCancelledError,
    StreamInactiveError,
    SubscriptionMessage,
)
from dapr.ext.workflow import DaprWorkflowClient
from dapr.ext.workflow._durabletask.internal import protos as workflow_protos
from dapr.proto import api_v1, appcallback_v1
from drasi_agent_router_contracts import AgentDelivery, to_wire
from grpc import RpcError, StatusCode
from pytest_mock import MockerFixture

from dapr_agents.ext.drasi import delivery
from dapr_agents.ext.drasi._interfaces import AdmissionHandler, IntentStoreError
from dapr_agents.ext.drasi._models import (
    AdmissionResult,
    Discard,
    IntentDocument,
    Poison,
    ResolvedDrasiConfig,
    Retry,
    SchedulingInput,
    SubscriptionScope,
)
from dapr_agents.ext.drasi.admission import DrasiAdmissionHandler
from dapr_agents.ext.drasi.delivery import (
    DrasiDeliveryError,
    _handle_message,
    subscribe_drasi_inbox,
)

from .fakes import InMemoryIntentRepository


class _RpcFailure(RpcError):
    def __init__(self, status: StatusCode) -> None:
        super().__init__("ROW_SENTINEL: already exists")
        self._status = status

    def code(self) -> StatusCode:
        return self._status

    def details(self) -> str:
        return "ROW_SENTINEL: already exists"


class _Stream:
    def __init__(self) -> None:
        self.messages: Queue[SubscriptionMessage | Exception | None] = Queue()
        self.responses: Queue[tuple[SubscriptionMessage, TopicEventResponseStatus]] = (
            Queue()
        )
        self.closed = Event()
        self.close_count = 0

    def __iter__(self) -> Iterator[SubscriptionMessage | None]:
        return self

    def __next__(self) -> SubscriptionMessage | None:
        message = self.messages.get(timeout=5)
        if isinstance(message, Exception):
            raise message
        return message

    def respond(
        self, message: SubscriptionMessage, status: TopicEventResponseStatus
    ) -> None:
        self.responses.put((message, status))

    def close(self) -> None:
        if not self.closed.is_set():
            self.close_count += 1
            self.closed.set()
            self.messages.put(StreamCancelledError())


@pytest.fixture
def config(scope: SubscriptionScope) -> ResolvedDrasiConfig:
    return ResolvedDrasiConfig(
        scope=scope,
        pubsub_name="application-pubsub",
        state_store_name="intent-store",
        workflow_name="CheckoutSRE_agent_workflow",
        dapr_http_port=3500,
    )


@pytest.fixture
def scheduling() -> SchedulingInput:
    return SchedulingInput(
        instance_id="drasi-55cc144e3197a732f9643970584a69022b0f4ed555a5a204ef57e4af6a827e67",
        event_id="drasi:v1:service-errors:42:i:0",
        task="Handle ROW_SENTINEL with INSTRUCTION_SENTINEL.",
    )


@pytest.fixture
def admission(scheduling: SchedulingInput, mocker: MockerFixture) -> Mock:
    result = mocker.Mock(spec=AdmissionHandler)
    result.admit.return_value = scheduling
    return result


@pytest.fixture
def workflow(scheduling: SchedulingInput, mocker: MockerFixture) -> MagicMock:
    result = mocker.create_autospec(DaprWorkflowClient, instance=True)
    result.schedule_new_workflow.return_value = scheduling.instance_id
    return result


@pytest.fixture
def client(mocker: MockerFixture) -> MagicMock:
    return mocker.create_autospec(DaprClient, instance=True)


@pytest.fixture
def stream(client: MagicMock) -> _Stream:
    result = _Stream()
    client.subscribe.return_value = result
    return result


@pytest.fixture
def handle(
    config: ResolvedDrasiConfig, admission: Mock, workflow: MagicMock
) -> Callable[[SubscriptionMessage], TopicEventResponse]:
    return partial(
        _handle_message,
        admission=admission,
        workflow_client=workflow,
        workflow_name=config.workflow_name,
    )


@pytest.fixture
def deeply_nested_json() -> bytes:
    # Newer Python decoders use a C-stack limit, not sys.getrecursionlimit().
    for depth in (2048, 4096, 8192, 16384, 32768, 65536):
        text = "[" * depth + "0" + "]" * depth
        try:
            json.loads(text)
        except RecursionError:
            return b"[" * (depth * 2) + b"0" + b"]" * (depth * 2)
    pytest.skip("This decoder has no nesting limit within the bounded probe.")


def _request(
    event: AgentDelivery,
    *,
    data: bytes | None = None,
    content_type: str = "application/json",
    publication_id: str = "outer-publication-id",
) -> appcallback_v1.TopicEventRequest:
    return appcallback_v1.TopicEventRequest(
        id=publication_id,
        source="router-application",
        type="com.dapr.event.sent",
        spec_version="1.0",
        pubsub_name="publisher-pubsub",
        topic="derived-inbox",
        data_content_type=content_type,
        data=json.dumps(to_wire(event)).encode("utf-8") if data is None else data,
    )


def _message(event: AgentDelivery) -> SubscriptionMessage:
    return SubscriptionMessage(_request(event))


@pytest.mark.parametrize(
    ("result", "expected"),
    (
        (Discard(reason="no_intent"), TopicEventResponseStatus.success),
        (Discard(reason="unavailable"), TopicEventResponseStatus.success),
        (Discard(reason="pending_unsubscribe"), TopicEventResponseStatus.success),
        (Discard(reason="stale_incarnation"), TopicEventResponseStatus.success),
        (Discard(reason="operation_excluded"), TopicEventResponseStatus.success),
        (Retry(reason="pending_subscription"), TopicEventResponseStatus.retry),
        (Retry(reason="state_unavailable"), TopicEventResponseStatus.retry),
        (Retry(reason="state_corrupt"), TopicEventResponseStatus.retry),
        (Retry(reason="unsupported_state_version"), TopicEventResponseStatus.retry),
        (Poison(reason="invalid_delivery"), TopicEventResponseStatus.drop),
        (Poison(reason="router_mismatch"), TopicEventResponseStatus.drop),
    ),
)
def test_admission_dispositions_do_not_schedule(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    admission: Mock,
    workflow: MagicMock,
    insert_delivery: AgentDelivery,
    result: AdmissionResult,
    expected: TopicEventResponseStatus,
) -> None:
    admission.admit.return_value = result

    assert handle(_message(insert_delivery)).status == expected
    workflow.schedule_new_workflow.assert_not_called()


def test_schedules_only_the_frozen_task_with_registered_name_and_instance_id(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    config: ResolvedDrasiConfig,
    admission: Mock,
    workflow: MagicMock,
    scheduling: SchedulingInput,
    insert_delivery: AgentDelivery,
) -> None:
    response = handle(_message(insert_delivery))

    assert response.status == TopicEventResponseStatus.success
    admission.admit.assert_called_once_with(to_wire(insert_delivery))
    workflow.schedule_new_workflow.assert_called_once_with(
        config.workflow_name,
        input={"task": scheduling.task},
        instance_id=scheduling.instance_id,
    )
    workflow.wait_for_workflow_start.assert_not_called()
    workflow.wait_for_workflow_completion.assert_not_called()
    workflow.get_workflow_state.assert_not_called()


@pytest.mark.parametrize(
    "content_type",
    (
        "application/json",
        "application/json; charset=utf-8",
        "Application/JSON",
        "application/vnd.drasi+json",
    ),
)
def test_reads_sdk_data_bytes_without_relying_on_its_eager_parser(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    admission: Mock,
    insert_delivery: AgentDelivery,
    content_type: str,
) -> None:
    message = SubscriptionMessage(_request(insert_delivery, content_type=content_type))

    assert handle(message).status == TopicEventResponseStatus.success
    admission.admit.assert_called_once_with(to_wire(insert_delivery))


@pytest.mark.parametrize(
    "content_type", ("text/plain", "application/octet-stream", "", "text/json")
)
def test_unsupported_content_types_request_dead_letters_without_admission(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    admission: Mock,
    workflow: MagicMock,
    insert_delivery: AgentDelivery,
    content_type: str,
) -> None:
    message = SubscriptionMessage(_request(insert_delivery, content_type=content_type))

    assert handle(message).status == TopicEventResponseStatus.drop
    admission.admit.assert_not_called()
    workflow.schedule_new_workflow.assert_not_called()


@pytest.mark.parametrize("data", (b"", b"{", b"\xff", b"ROW_SENTINEL"))
def test_malformed_json_requests_dead_letters_without_admission(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    admission: Mock,
    workflow: MagicMock,
    insert_delivery: AgentDelivery,
    data: bytes,
) -> None:
    message = SubscriptionMessage(_request(insert_delivery, data=data))

    assert handle(message).status == TopicEventResponseStatus.drop
    admission.admit.assert_not_called()
    workflow.schedule_new_workflow.assert_not_called()


def test_excessively_nested_json_is_poison_not_a_consumer_failure(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    admission: Mock,
    workflow: MagicMock,
    insert_delivery: AgentDelivery,
    deeply_nested_json: bytes,
) -> None:
    message = SubscriptionMessage(_request(insert_delivery, data=deeply_nested_json))

    assert handle(message).status == TopicEventResponseStatus.drop
    admission.admit.assert_not_called()
    workflow.schedule_new_workflow.assert_not_called()


def test_decoding_preserves_large_integer_sequences(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    admission: Mock,
    insert_delivery: AgentDelivery,
) -> None:
    data = to_wire(insert_delivery)
    data["event"]["seq"] = 2**64 - 1
    data["eventId"] = f"drasi:v1:service-errors:{2**64 - 1}:i:0"
    message = SubscriptionMessage(
        _request(insert_delivery, data=json.dumps(data).encode("utf-8"))
    )

    assert handle(message).status == TopicEventResponseStatus.success
    decoded = admission.admit.call_args.args[0]
    assert type(decoded["event"]["seq"]) is int
    assert decoded["event"]["seq"] == 2**64 - 1


@pytest.mark.parametrize("status", tuple(StatusCode))
def test_only_structured_duplicate_conflicts_acknowledge_scheduler_errors(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    workflow: MagicMock,
    insert_delivery: AgentDelivery,
    status: StatusCode,
) -> None:
    workflow.schedule_new_workflow.side_effect = _RpcFailure(status)
    expected = (
        TopicEventResponseStatus.success
        if status == StatusCode.ALREADY_EXISTS
        else TopicEventResponseStatus.retry
    )

    assert handle(_message(insert_delivery)).status == expected


@pytest.mark.parametrize(
    "error",
    (
        RuntimeError("already exists"),
        TimeoutError("ROW_SENTINEL"),
        RpcError("already exists"),
    ),
)
def test_ambiguous_scheduler_errors_are_never_assumed_delivered(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    workflow: MagicMock,
    insert_delivery: AgentDelivery,
    error: Exception,
) -> None:
    workflow.schedule_new_workflow.side_effect = error

    assert handle(_message(insert_delivery)).status == TopicEventResponseStatus.retry


@pytest.mark.parametrize("instance_id", (None, "", "another-instance"))
def test_mismatched_scheduler_confirmations_retry(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    workflow: MagicMock,
    insert_delivery: AgentDelivery,
    instance_id: str | None,
) -> None:
    workflow.schedule_new_workflow.return_value = instance_id

    assert handle(_message(insert_delivery)).status == TopicEventResponseStatus.retry


@pytest.mark.parametrize(
    "error",
    (
        RuntimeError("ROW_SENTINEL"),
        IntentStoreError("unavailable"),
        _RpcFailure(StatusCode.ALREADY_EXISTS),
    ),
)
def test_admission_errors_cannot_masquerade_as_scheduler_conflicts(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    admission: Mock,
    workflow: MagicMock,
    insert_delivery: AgentDelivery,
    error: Exception,
) -> None:
    admission.admit.side_effect = error

    assert handle(_message(insert_delivery)).status == TopicEventResponseStatus.retry
    workflow.schedule_new_workflow.assert_not_called()


def test_invalid_admission_result_is_visible_and_retries(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    admission: Mock,
    insert_delivery: AgentDelivery,
    caplog: pytest.LogCaptureFixture,
) -> None:
    admission.admit.return_value = None

    assert handle(_message(insert_delivery)).status == TopicEventResponseStatus.retry
    assert "invalid admission result" in caplog.text


def test_terminal_id_reuse_remains_possible_without_a_local_deduplication_cache(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    workflow: MagicMock,
    scheduling: SchedulingInput,
    insert_delivery: AgentDelivery,
) -> None:
    workflow.schedule_new_workflow.side_effect = (
        scheduling.instance_id,
        _RpcFailure(StatusCode.ALREADY_EXISTS),
        scheduling.instance_id,
    )
    message = _message(insert_delivery)

    assert handle(message).status == TopicEventResponseStatus.success
    assert handle(message).status == TopicEventResponseStatus.success
    assert handle(message).status == TopicEventResponseStatus.success
    assert workflow.schedule_new_workflow.call_count == 3
    workflow.get_workflow_state.assert_not_called()


def test_real_admission_ignores_outer_publication_identity_and_rejects_double_envelopes(
    config: ResolvedDrasiConfig,
    scope: SubscriptionScope,
    intent_document: IntentDocument,
    workflow: MagicMock,
    insert_delivery: AgentDelivery,
) -> None:
    repository = InMemoryIntentRepository(scope)
    repository.initialize(intent_document)
    original = repository.load()
    admit = DrasiAdmissionHandler(scope=scope, intents=repository)
    handler = partial(
        _handle_message,
        admission=admit,
        workflow_client=workflow,
        workflow_name=config.workflow_name,
    )
    for publication_id in ("first-publication", "retry-publication"):
        message = SubscriptionMessage(
            _request(insert_delivery, publication_id=publication_id)
        )
        assert handler(message).status == TopicEventResponseStatus.success
    first, second = workflow.schedule_new_workflow.call_args_list
    assert first == second

    outer = {"specversion": "1.0", "id": "outer", "data": to_wire(insert_delivery)}
    message = SubscriptionMessage(
        _request(insert_delivery, data=json.dumps(outer).encode("utf-8"))
    )
    assert handler(message).status == TopicEventResponseStatus.drop
    assert workflow.schedule_new_workflow.call_count == 2
    assert repository.load() == original


def test_subscribes_to_derived_inbox_and_dlt_and_owns_the_dapr_client(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
) -> None:
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )

    client.subscribe.assert_called_once_with(
        pubsub_name=config.pubsub_name,
        topic=config.scope.inbox_topic,
        dead_letter_topic=config.scope.dead_letter_topic,
    )
    close()
    close()
    assert stream.closed.is_set()
    assert stream.close_count == 1
    client.close.assert_called_once()
    client.publish_event.assert_not_called()
    workflow.close.assert_not_called()
    admission.admit.assert_not_called()


@pytest.mark.parametrize(
    ("outcome", "status"),
    (
        (Discard(reason="no_intent"), TopicEventResponseStatus.success),
        (Retry(reason="pending_subscription"), TopicEventResponseStatus.retry),
        (Poison(reason="invalid_delivery"), TopicEventResponseStatus.drop),
    ),
)
def test_consumer_sends_the_correct_transport_disposition(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
    insert_delivery: AgentDelivery,
    outcome: AdmissionResult,
    status: TopicEventResponseStatus,
) -> None:
    admission.admit.return_value = outcome
    message = _message(insert_delivery)
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )
    try:
        stream.messages.put(None)
        stream.messages.put(message)
        assert stream.responses.get(timeout=5) == (message, status)
    finally:
        close()
    workflow.schedule_new_workflow.assert_not_called()
    client.publish_event.assert_not_called()


def test_no_acknowledgement_before_scheduling_acceptance(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
    scheduling: SchedulingInput,
    insert_delivery: AgentDelivery,
) -> None:
    entered = Event()
    accepted = Event()

    def schedule(*args: object, **kwargs: object) -> str:
        entered.set()
        assert accepted.wait(timeout=5)
        return scheduling.instance_id

    workflow.schedule_new_workflow.side_effect = schedule
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )
    message = _message(insert_delivery)
    try:
        stream.messages.put(message)
        assert entered.wait(timeout=5)
        assert stream.responses.empty()
        accepted.set()
        assert stream.responses.get(timeout=5) == (
            message,
            TopicEventResponseStatus.success,
        )
    finally:
        accepted.set()
        close()
    workflow.wait_for_workflow_completion.assert_not_called()


def test_scheduler_failure_is_a_retry_on_the_live_consumer(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
    insert_delivery: AgentDelivery,
) -> None:
    workflow.schedule_new_workflow.side_effect = TimeoutError("INSTRUCTION_SENTINEL")
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )
    message = _message(insert_delivery)
    try:
        stream.messages.put(message)
        assert stream.responses.get(timeout=5) == (
            message,
            TopicEventResponseStatus.retry,
        )
    finally:
        close()


def test_consumer_dead_letters_deep_json_and_continues_with_valid_deliveries(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
    insert_delivery: AgentDelivery,
    deeply_nested_json: bytes,
) -> None:
    poison = SubscriptionMessage(
        _request(
            insert_delivery,
            data=deeply_nested_json,
            publication_id="poison-publication",
        )
    )
    valid = _message(insert_delivery)
    close = subscribe_drasi_inbox(
        config=config,
        admission=admission,
        dapr_client=client,
        workflow_client=workflow,
    )
    try:
        stream.messages.put(poison)
        stream.messages.put(valid)
        assert stream.responses.get(timeout=5) == (
            poison,
            TopicEventResponseStatus.drop,
        )
        assert stream.responses.get(timeout=5) == (
            valid,
            TopicEventResponseStatus.success,
        )
        admission.admit.assert_called_once_with(to_wire(insert_delivery))
        assert not stream.closed.is_set()
    finally:
        close()


@pytest.mark.parametrize(
    "error",
    (
        StreamCancelledError("ROW_SENTINEL"),
        StreamInactiveError("ROW_SENTINEL"),
        RuntimeError("ROW_SENTINEL"),
        StopIteration(),
    ),
)
def test_unexpected_stream_termination_is_logged_without_blocking_cleanup(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )
    stream.messages.put(error)
    assert stream.closed.wait(timeout=5)

    close()
    close()
    assert any(record.levelno >= logging.ERROR for record in caplog.records)
    assert "SENTINEL" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_unexpected_subscription_exception_closes_the_consumer(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
    insert_delivery: AgentDelivery,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mocker.patch.object(stream, "respond", side_effect=RuntimeError("ROW_SENTINEL"))
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )
    stream.messages.put(_message(insert_delivery))
    assert stream.closed.wait(timeout=5)

    close()
    assert "Drasi inbox consumer failed" in caplog.text
    assert stream.responses.empty()


def test_subscription_startup_failure_does_not_return_a_successful_closer(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client.subscribe.side_effect = RuntimeError("ROW_SENTINEL")

    with pytest.raises(DrasiDeliveryError, match="Could not subscribe"):
        subscribe_drasi_inbox(
            config=config,
            admission=admission,
            dapr_client=client,
            workflow_client=workflow,
        )
    assert "SENTINEL" not in caplog.text
    client.close.assert_not_called()
    workflow.close.assert_not_called()


def test_thread_startup_failure_closes_the_opened_subscription(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
    mocker: MockerFixture,
) -> None:
    thread = mocker.patch.object(delivery, "Thread", autospec=True).return_value
    thread.start.side_effect = RuntimeError("ROW_SENTINEL")

    with pytest.raises(DrasiDeliveryError, match="Could not start"):
        subscribe_drasi_inbox(
            config=config,
            admission=admission,
            dapr_client=client,
            workflow_client=workflow,
        )
    assert stream.closed.is_set()
    assert stream.close_count == 1


def test_close_attempts_transport_and_subscription_independently(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original_close = stream.close
    attempts = 0

    def close_stream() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("ROW_SENTINEL")
        original_close()

    mocker.patch.object(stream, "close", side_effect=close_stream)
    client.close.side_effect = stream.close
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )
    with pytest.raises(DrasiDeliveryError, match="Dapr client"):
        close()
    assert not close.is_closed
    assert close.is_stopped
    close()
    assert close.is_closed
    assert stream.closed.is_set()
    assert attempts >= 2
    assert stream.close_count == 1
    assert "SENTINEL" not in caplog.text


def test_background_cleanup_failure_is_logged_and_owner_can_finish_cleanup(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempted = Event()
    original_close = stream.close

    def close_stream() -> None:
        if not attempted.is_set():
            attempted.set()
            raise RuntimeError("ROW_SENTINEL")
        original_close()

    mocker.patch.object(stream, "close", side_effect=close_stream)
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )
    stream.messages.put(StopIteration())
    assert attempted.wait(timeout=5)

    close()
    assert "Drasi inbox stream close failed" in caplog.text
    assert stream.closed.is_set()


def test_worker_cannot_join_itself(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
    insert_delivery: AgentDelivery,
) -> None:
    errors: Queue[DrasiDeliveryError] = Queue()
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )

    def admit(data: object) -> Discard:
        try:
            close()
        except DrasiDeliveryError as error:
            errors.put(error)
        return Discard(reason="no_intent")

    admission.admit.side_effect = admit
    try:
        stream.messages.put(_message(insert_delivery))
        assert "cannot close itself" in str(errors.get(timeout=5))
    finally:
        close()
    assert stream.responses.empty()


def test_shutdown_does_not_start_work_for_an_already_received_message(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    insert_delivery: AgentDelivery,
) -> None:
    received = Event()
    proceed = Event()

    class PausedStream(_Stream):
        def __next__(self) -> SubscriptionMessage | None:
            message = super().__next__()
            received.set()
            assert proceed.wait(timeout=5)
            return message

        def close(self) -> None:
            super().close()
            proceed.set()

    stream = PausedStream()
    client.subscribe.return_value = stream
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )
    try:
        stream.messages.put(_message(insert_delivery))
        assert received.wait(timeout=5)
    finally:
        close()
    admission.admit.assert_not_called()
    workflow.schedule_new_workflow.assert_not_called()
    assert stream.responses.empty()


def test_shutdown_timeout_is_explicit_and_inflight_work_is_not_acknowledged(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    stream: _Stream,
    scheduling: SchedulingInput,
    insert_delivery: AgentDelivery,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = Event()
    accepted = Event()

    def schedule(*args: object, **kwargs: object) -> str:
        entered.set()
        assert accepted.wait(timeout=5)
        return scheduling.instance_id

    workflow.schedule_new_workflow.side_effect = schedule
    monkeypatch.setattr(delivery, "_SHUTDOWN_TIMEOUT_SECONDS", 0.01)
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )
    try:
        stream.messages.put(_message(insert_delivery))
        assert entered.wait(timeout=5)
        with pytest.raises(DrasiDeliveryError, match="did not stop in time"):
            close()
        assert not close.is_stopped
        assert not close.is_closed
    finally:
        monkeypatch.setattr(delivery, "_SHUTDOWN_TIMEOUT_SECONDS", 5)
        accepted.set()
        close()
    assert close.is_stopped
    assert close.is_closed
    assert stream.responses.empty()


def test_message_error_logs_hide_payloads_and_raw_exception_text(
    handle: Callable[[SubscriptionMessage], TopicEventResponse],
    admission: Mock,
    workflow: MagicMock,
    scheduling: SchedulingInput,
    insert_delivery: AgentDelivery,
    caplog: pytest.LogCaptureFixture,
) -> None:
    message = _message(insert_delivery)
    with caplog.at_level(logging.DEBUG):
        admission.admit.side_effect = RuntimeError("ROW_SENTINEL")
        assert handle(message).status == TopicEventResponseStatus.retry
        admission.admit.side_effect = None
        workflow.schedule_new_workflow.side_effect = _RpcFailure(StatusCode.UNKNOWN)
        assert handle(message).status == TopicEventResponseStatus.retry
        workflow.schedule_new_workflow.side_effect = RuntimeError(
            "INSTRUCTION_SENTINEL"
        )
        assert handle(message).status == TopicEventResponseStatus.retry
        admission.admit.return_value = Poison(reason="invalid_delivery")
        assert handle(message).status == TopicEventResponseStatus.drop
    assert "SENTINEL" not in caplog.text
    assert scheduling.task not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


class _NativeStream:
    def __init__(self) -> None:
        self.messages: Queue[appcallback_v1.TopicEventRequest | _RpcFailure] = Queue()
        self.initial = True
        self.cancelled = False

    def __iter__(self) -> Iterator[api_v1.SubscribeTopicEventsResponseAlpha1]:
        return self

    def __next__(self) -> api_v1.SubscribeTopicEventsResponseAlpha1:
        if self.initial:
            self.initial = False
            return api_v1.SubscribeTopicEventsResponseAlpha1(
                initial_response=api_v1.SubscribeTopicEventsResponseInitialAlpha1()
            )
        message = self.messages.get(timeout=5)
        if isinstance(message, _RpcFailure):
            raise message
        return api_v1.SubscribeTopicEventsResponseAlpha1(event_message=message)

    def cancel(self) -> bool:
        self.cancelled = True
        self.messages.put(_RpcFailure(StatusCode.CANCELLED))
        return True


def test_shutdown_closes_the_sdk_stream_reopened_during_an_inflight_reconnect(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    mocker: MockerFixture,
) -> None:
    original_stream = _NativeStream()
    replacement_stream = _NativeStream()
    stub = mocker.Mock()
    stub.SubscribeTopicEventsAlpha1.side_effect = (original_stream, replacement_stream)
    subscription = Subscription(
        stub,
        config.pubsub_name,
        config.scope.inbox_topic,
        dead_letter_topic=config.scope.dead_letter_topic,
    )
    subscription.start()
    client.subscribe.return_value = subscription
    reconnect_waiting = Event()
    allow_reconnect = Event()

    def wait_for_sidecar() -> None:
        reconnect_waiting.set()
        assert allow_reconnect.wait(timeout=5)

    mocker.patch(
        "dapr.clients.grpc.subscription.DaprHealth.wait_for_sidecar",
        side_effect=wait_for_sidecar,
    )
    original_close = subscription.close

    def close_stream() -> None:
        original_close()
        if reconnect_waiting.is_set():
            allow_reconnect.set()

    mocker.patch.object(subscription, "close", side_effect=close_stream)
    close = subscribe_drasi_inbox(
        config=config,
        admission=admission,
        dapr_client=client,
        workflow_client=workflow,
    )
    try:
        original_stream.messages.put(_RpcFailure(StatusCode.UNAVAILABLE))
        assert reconnect_waiting.wait(timeout=5)
        close()
        assert replacement_stream.cancelled
        assert not subscription._is_stream_active()
        admission.admit.assert_not_called()
        workflow.schedule_new_workflow.assert_not_called()
    finally:
        allow_reconnect.set()
        original_close()
        close()


@pytest.mark.parametrize(
    ("outcome", "status"),
    (
        (Discard(reason="no_intent"), appcallback_v1.TopicEventResponse.SUCCESS),
        (Retry(reason="state_unavailable"), appcallback_v1.TopicEventResponse.RETRY),
        (Poison(reason="invalid_delivery"), appcallback_v1.TopicEventResponse.DROP),
    ),
)
def test_native_sdk_subscription_carries_dlt_and_the_outer_ack_id(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    insert_delivery: AgentDelivery,
    mocker: MockerFixture,
    outcome: AdmissionResult,
    status: int,
) -> None:
    native_stream = _NativeStream()
    stub = mocker.Mock()
    stub.SubscribeTopicEventsAlpha1.return_value = native_stream
    native_subscription = Subscription(
        stub,
        config.pubsub_name,
        config.scope.inbox_topic,
        dead_letter_topic=config.scope.dead_letter_topic,
    )
    native_subscription.start()
    requests = stub.SubscribeTopicEventsAlpha1.call_args.args[0]
    initial = next(requests).initial_request
    client.subscribe.return_value = native_subscription
    admission.admit.return_value = outcome
    close = subscribe_drasi_inbox(
        config=config, admission=admission, dapr_client=client, workflow_client=workflow
    )
    try:
        native_stream.messages.put(_request(insert_delivery))
        response = native_subscription._send_queue.get(timeout=5).event_processed
        assert initial.pubsub_name == config.pubsub_name
        assert initial.topic == config.scope.inbox_topic
        assert initial.dead_letter_topic == config.scope.dead_letter_topic
        assert response.id == "outer-publication-id"
        assert response.id != insert_delivery.eventId
        assert response.status.status == status
    finally:
        close()
    assert native_stream.cancelled
    client.publish_event.assert_not_called()


def test_native_sdk_inactive_response_failure_surfaces_on_the_next_read(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    scheduling: SchedulingInput,
    insert_delivery: AgentDelivery,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    native_stream = _NativeStream()
    stub = mocker.Mock()
    stub.SubscribeTopicEventsAlpha1.return_value = native_stream
    subscription = Subscription(
        stub,
        config.pubsub_name,
        config.scope.inbox_topic,
        dead_letter_topic=config.scope.dead_letter_topic,
    )
    subscription.start()
    client.subscribe.return_value = subscription
    threads: list[Thread] = []

    def make_thread(*, target: Callable[[], None], name: str, daemon: bool) -> Thread:
        thread = Thread(target=target, name=name, daemon=daemon)
        threads.append(thread)
        return thread

    mocker.patch.object(delivery, "Thread", side_effect=make_thread)

    def schedule(*args: object, **kwargs: object) -> str:
        subscription.close()
        return scheduling.instance_id

    workflow.schedule_new_workflow.side_effect = schedule
    close = subscribe_drasi_inbox(
        config=config,
        admission=admission,
        dapr_client=client,
        workflow_client=workflow,
    )
    native_stream.messages.put(_request(insert_delivery))
    threads[0].join(timeout=5)

    assert not threads[0].is_alive()
    assert subscription._send_queue.empty()
    assert "Can't send message on inactive stream" in caplog.text
    close()
    assert "Drasi inbox stream stopped" in caplog.text
    assert native_stream.cancelled


def test_native_sdk_lost_response_does_not_suppress_redelivery(
    config: ResolvedDrasiConfig,
    admission: Mock,
    client: MagicMock,
    workflow: MagicMock,
    scheduling: SchedulingInput,
    insert_delivery: AgentDelivery,
    mocker: MockerFixture,
) -> None:
    native_stream = _NativeStream()
    stub = mocker.Mock()
    stub.SubscribeTopicEventsAlpha1.return_value = native_stream
    subscription = Subscription(
        stub,
        config.pubsub_name,
        config.scope.inbox_topic,
        dead_letter_topic=config.scope.dead_letter_topic,
    )
    subscription.start()
    client.subscribe.return_value = subscription
    refused = Event()
    original_put = subscription._send_queue.put

    def put(response: api_v1.SubscribeTopicEventsRequestAlpha1) -> None:
        if not refused.is_set():
            refused.set()
            raise RuntimeError("simulated response enqueue failure")
        original_put(response)

    mocker.patch.object(subscription._send_queue, "put", side_effect=put)
    workflow.schedule_new_workflow.side_effect = (
        scheduling.instance_id,
        _RpcFailure(StatusCode.ALREADY_EXISTS),
    )
    close = subscribe_drasi_inbox(
        config=config,
        admission=admission,
        dapr_client=client,
        workflow_client=workflow,
    )
    try:
        native_stream.messages.put(_request(insert_delivery))
        assert refused.wait(timeout=5)
        assert subscription._send_queue.empty()
        assert workflow.schedule_new_workflow.call_count == 1

        native_stream.messages.put(_request(insert_delivery))
        response = subscription._send_queue.get(timeout=5).event_processed
        assert response.id == "outer-publication-id"
        assert response.status.status == appcallback_v1.TopicEventResponse.SUCCESS
        assert workflow.schedule_new_workflow.call_count == 2
        assert admission.admit.call_count == 2
    finally:
        close()


def test_native_workflow_client_sends_the_expected_start_request(
    config: ResolvedDrasiConfig,
    admission: Mock,
    scheduling: SchedulingInput,
    insert_delivery: AgentDelivery,
    mocker: MockerFixture,
) -> None:
    channel = mocker.patch(
        "dapr.ext.workflow._durabletask.internal.shared.get_grpc_channel"
    ).return_value
    stub = mocker.patch(
        "dapr.ext.workflow._durabletask.client.stubs.TaskHubSidecarServiceStub"
    ).return_value
    stub.StartInstance.return_value = workflow_protos.CreateInstanceResponse(
        instanceId=scheduling.instance_id
    )
    workflow = DaprWorkflowClient(host="127.0.0.1", port="50001")
    try:
        result = _handle_message(
            _message(insert_delivery),
            admission=admission,
            workflow_client=workflow,
            workflow_name=config.workflow_name,
        )
        assert result.status == TopicEventResponseStatus.success
        request = stub.StartInstance.call_args.args[0]
        assert request.name == config.workflow_name
        assert request.instanceId == scheduling.instance_id
        assert json.loads(request.input.value) == {"task": scheduling.task}
        stub.WaitForInstanceStart.assert_not_called()
        stub.WaitForInstanceCompletion.assert_not_called()
        channel.close.assert_not_called()
    finally:
        workflow.close()
