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

import asyncio
import logging
import time
import pytest
from typing import Union, Optional, List
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from pydantic import BaseModel, Field

from dapr_agents.workflow.decorators.decorators import message_router
from dapr_agents.workflow.utils.routers import (
    extract_message_models,
    extract_cloudevent_data,
    validate_message_model,
    parse_cloudevent,
)
from dapr_agents.workflow.utils.registration import register_message_routes
from dapr_agents.workflow.utils.subscription import (
    DELIVERY_MODE_ASYNC,
    DELIVERY_MODE_SYNC,
    METADATA_KEY,
    STATUS_DROP,
    STATUS_RETRY,
    STATUS_SUCCESS,
    MessageContext,
    MessageRouteBinding,
    TTLDedupeBackend,
    WorkflowStatus,
    _safe_map,
    validate_hooks,
    _attach_metadata_to_payload,
    _build_binding_schema_pairs,
    _cancel_tasks,
    _filter_accepts,
    _group_bindings_by_topic,
    _log_workflow_outcome,
    _normalize_status,
    _order_pairs_by_cloudevent_type,
    _resolve_event_loop,
    _serialize_workflow_input,
    _shutdown_thread,
    _validate_dead_letter_topics,
    _validate_delivery_mode,
    _warn_unreachable_bindings,
)
from dapr.clients.grpc._response import TopicEventResponseStatus
from dapr_agents.types.message import EventMessageMetadata


_PATCH_TARGET = "dapr_agents.workflow.utils.registration.default_dapr_client_factory"


def create_mock_dapr_client(pubsub_names: List[str]) -> MagicMock:
    """
    Create a mock DaprClient with specified pubsub components registered.

    Args:
        pubsub_names: List of pubsub component names to register in the mock.

    Returns:
        A MagicMock configured to return the pubsub components in get_metadata.
    """
    mock_client = MagicMock()
    mock_sub = MagicMock()
    mock_sub.__iter__.return_value = iter([])
    mock_client.subscribe.return_value = mock_sub

    # Set up get_metadata to return the pubsub components
    mock_metadata = MagicMock()
    components = []
    for name in pubsub_names:
        component = MagicMock()
        component.type = "pubsub.redis"
        component.name = name
        components.append(component)
    mock_metadata.registered_components = components
    mock_client.get_metadata.return_value = mock_metadata

    # Support context manager usage (with DaprClient() as client:)
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    return mock_client


# Test Models
class OrderCreated(BaseModel):
    """Test Pydantic model for order creation events."""

    order_id: str = Field(..., description="Unique order identifier")
    amount: float = Field(..., description="Order amount")
    customer: str = Field(..., description="Customer name")


class OrderCancelled(BaseModel):
    """Test Pydantic model for order cancellation events."""

    order_id: str = Field(..., description="Order ID to cancel")
    reason: str = Field(..., description="Cancellation reason")


@dataclass
class ShipmentCreated:
    """Test dataclass for shipment events."""

    shipment_id: str
    order_id: str
    carrier: str


# Tests for extract_message_models utility


def test_extract_message_models_single_class():
    """Test extracting a single model class."""
    models = extract_message_models(OrderCreated)
    assert models == [OrderCreated]


def test_extract_message_models_union():
    """Test extracting models from Union type hint."""
    models = extract_message_models(Union[OrderCreated, OrderCancelled])
    assert set(models) == {OrderCreated, OrderCancelled}


def test_extract_message_models_optional():
    """Test extracting models from Optional type hint (filters out None)."""
    models = extract_message_models(Optional[OrderCreated])
    assert models == [OrderCreated]


def test_extract_message_models_pipe_union():
    """Test extracting models from pipe union syntax (Python 3.10+)."""
    # Note: This test requires Python 3.10+ for the | syntax
    try:
        hint = eval("OrderCreated | OrderCancelled")
        models = extract_message_models(hint)
        assert set(models) == {OrderCreated, OrderCancelled}
    except SyntaxError:
        pytest.skip("Python 3.10+ required for pipe union syntax")


def test_extract_message_models_none_input():
    """Test extracting models from None returns empty list."""
    models = extract_message_models(None)
    assert models == []


def test_extract_message_models_non_class():
    """Test extracting models from non-class type returns empty list."""
    models = extract_message_models("not a class")
    assert models == []


# Tests for message_router decorator


def test_message_router_requires_message_model():
    """Test that message_router raises TypeError when message_model is missing and can't be inferred."""
    with pytest.raises(
        TypeError,
        match="`@message_router` requires `message_model`",
    ):

        @message_router(pubsub="messagepubsub", topic="orders")
        def handler(data: OrderCreated):  # Wrong parameter name, can't infer
            pass


def test_message_router_requires_type_hint():
    """Test that message_router raises TypeError when message parameter has no type hint."""
    with pytest.raises(TypeError, match="`@message_router` requires `message_model`"):

        @message_router(pubsub="messagepubsub", topic="orders")
        def handler(message):  # No type hint
            pass


def test_message_router_unsupported_type():
    """Test that message_router raises TypeError for unsupported message types."""
    with pytest.raises(TypeError, match="Unsupported model type"):

        @message_router(pubsub="messagepubsub", topic="orders")
        def handler(message: str):  # str is not a supported model
            pass


def test_message_router_basic_decoration():
    """Test basic message_router decoration with single model."""

    @message_router(pubsub="messagepubsub", topic="orders.created")
    def handle_order(message: OrderCreated):
        return message.order_id

    # Check metadata attributes
    assert hasattr(handle_order, "_is_message_handler")
    assert handle_order._is_message_handler is True
    assert hasattr(handle_order, "_message_router_data")

    data = handle_order._message_router_data
    assert data["pubsub"] == "messagepubsub"
    assert data["topic"] == "orders.created"
    assert data["dead_letter_topic"] == "orders.created_DEAD"
    assert data["is_broadcast"] is False
    assert OrderCreated in data["message_schemas"]
    assert "OrderCreated" in data["message_types"]


def test_message_router_with_dead_letter_topic():
    """Test message_router with custom dead letter topic."""

    @message_router(
        pubsub="messagepubsub",
        topic="orders.created",
        dead_letter_topic="orders.failed",
    )
    def handle_order(message: OrderCreated):
        pass

    data = handle_order._message_router_data
    assert data["dead_letter_topic"] == "orders.failed"


def test_message_router_with_broadcast():
    """Test message_router with broadcast flag."""

    @message_router(pubsub="messagepubsub", topic="notifications", broadcast=True)
    def handle_notification(message: OrderCreated):
        pass

    data = handle_notification._message_router_data
    assert data["is_broadcast"] is True


def test_message_router_union_types():
    """Test message_router with Union of multiple message types."""

    @message_router(pubsub="messagepubsub", topic="order.events")
    def handle_order_event(message: Union[OrderCreated, OrderCancelled]):
        pass

    data = handle_order_event._message_router_data
    assert set(data["message_schemas"]) == {OrderCreated, OrderCancelled}
    assert set(data["message_types"]) == {"OrderCreated", "OrderCancelled"}


def test_message_router_dataclass_model():
    """Test message_router with dataclass model."""

    @message_router(pubsub="messagepubsub", topic="shipments")
    def handle_shipment(message: ShipmentCreated):
        pass

    data = handle_shipment._message_router_data
    assert ShipmentCreated in data["message_schemas"]
    assert "ShipmentCreated" in data["message_types"]


def test_message_router_preserves_function_metadata():
    """Test that message_router preserves function name and docstring."""

    @message_router(pubsub="messagepubsub", topic="orders")
    def my_handler(message: OrderCreated):
        """Handler for order created events."""
        return "processed"

    assert my_handler.__name__ == "my_handler"
    assert my_handler.__doc__ == "Handler for order created events."


def test_message_router_function_still_callable():
    """Test that decorated function is still callable."""

    @message_router(pubsub="messagepubsub", topic="orders")
    def handle_order(message: OrderCreated):
        return f"Processed order {message.order_id}"

    # Function should still be callable with the right arguments
    test_order = OrderCreated(order_id="123", amount=99.99, customer="Alice")
    result = handle_order(test_order)
    assert result == "Processed order 123"


# Tests for validate_message_model utility


def test_validate_message_model_pydantic():
    """Test validating data against Pydantic model."""
    event_data = {"order_id": "123", "amount": 99.99, "customer": "Alice"}
    result = validate_message_model(OrderCreated, event_data)

    assert isinstance(result, OrderCreated)
    assert result.order_id == "123"
    assert result.amount == 99.99
    assert result.customer == "Alice"


def test_validate_message_model_dataclass():
    """Test validating data against dataclass model."""
    event_data = {"shipment_id": "S123", "order_id": "O456", "carrier": "FedEx"}
    result = validate_message_model(ShipmentCreated, event_data)

    assert isinstance(result, ShipmentCreated)
    assert result.shipment_id == "S123"
    assert result.order_id == "O456"
    assert result.carrier == "FedEx"


def test_validate_message_model_dict():
    """Test validating data against dict model (passthrough)."""
    event_data = {"key": "value", "number": 42}
    result = validate_message_model(dict, event_data)

    assert result == event_data
    assert isinstance(result, dict)


def test_validate_message_model_validation_error():
    """Test that validation errors are raised properly."""
    # Missing required field
    event_data = {"order_id": "123"}  # Missing 'amount' and 'customer'

    with pytest.raises(ValueError, match="Message validation failed"):
        validate_message_model(OrderCreated, event_data)


def test_validate_message_model_unsupported_type():
    """Test that unsupported model types raise TypeError."""

    class UnsupportedModel:
        pass

    with pytest.raises(TypeError, match="Unsupported model type"):
        validate_message_model(UnsupportedModel, {})


# Tests for extract_cloudevent_data utility


def test_extract_cloudevent_data_from_dict():
    """Test extracting CloudEvent data from dict envelope."""
    message = {
        "id": "event-123",
        "source": "order-service",
        "type": "order.created",
        "datacontenttype": "application/json",
        "data": {"order_id": "123", "amount": 99.99, "customer": "Alice"},
        "topic": "orders",
        "pubsubname": "messagepubsub",
        "specversion": "1.0",
    }

    event_data, metadata = extract_cloudevent_data(message)

    assert event_data == {"order_id": "123", "amount": 99.99, "customer": "Alice"}
    assert metadata["id"] == "event-123"
    assert metadata["source"] == "order-service"
    assert metadata["type"] == "order.created"
    assert metadata["topic"] == "orders"
    assert metadata["pubsubname"] == "messagepubsub"


def test_extract_cloudevent_data_from_dict_already_parsed():
    """Test extracting CloudEvent when data is already a dict."""
    message = {
        "id": "event-123",
        "data": {"key": "value"},  # Already a dict
        "datacontenttype": "application/json",
    }

    event_data, metadata = extract_cloudevent_data(message)
    assert event_data == {"key": "value"}


def test_extract_cloudevent_data_from_bytes():
    """Test extracting CloudEvent data from bytes payload."""
    import json

    payload = json.dumps({"order_id": "123", "amount": 99.99}).encode("utf-8")
    event_data, metadata = extract_cloudevent_data(payload)

    assert event_data == {"order_id": "123", "amount": 99.99}
    assert metadata["datacontenttype"] == "application/json"


def test_extract_cloudevent_data_from_str():
    """Test extracting CloudEvent data from string payload."""
    import json

    payload = json.dumps({"order_id": "123", "amount": 99.99})
    event_data, metadata = extract_cloudevent_data(payload)

    assert event_data == {"order_id": "123", "amount": 99.99}
    assert metadata["datacontenttype"] == "application/json"


def test_extract_cloudevent_data_from_subscription_message():
    """Test extracting CloudEvent from Dapr SubscriptionMessage."""
    import json
    from unittest.mock import MagicMock as MockClass

    mock_message = MockClass()
    mock_message.id.return_value = "event-456"
    mock_message.source.return_value = "test-service"
    mock_message.type.return_value = "test.event"
    mock_message.data_content_type.return_value = "application/json"
    mock_message.data.return_value = json.dumps({"key": "value"}).encode("utf-8")
    mock_message.topic.return_value = "test-topic"
    mock_message.pubsub_name.return_value = "test-pubsub"
    mock_message.spec_version.return_value = "1.0"
    mock_message.extensions.return_value = {}

    event_data, metadata = extract_cloudevent_data(mock_message)

    assert event_data == {"key": "value"}
    assert metadata["id"] == "event-456"
    assert metadata["source"] == "test-service"
    assert metadata["topic"] == "test-topic"


def test_extract_cloudevent_data_unsupported_type():
    """Test that unsupported message types raise ValueError."""
    with pytest.raises(ValueError, match="Unexpected message type"):
        extract_cloudevent_data(12345)  # int is not supported


def test_extract_cloudevent_data_non_dict_data():
    """Test handling non-dict event data (e.g., array)."""
    message = {
        "id": "event-123",
        "data": [1, 2, 3],  # Array data
        "datacontenttype": "application/json",
    }

    event_data, metadata = extract_cloudevent_data(message)
    assert event_data == [1, 2, 3]
    assert isinstance(event_data, list)


# Tests for parse_cloudevent utility


def test_parse_cloudevent_with_pydantic_model():
    """Test parsing CloudEvent with Pydantic model validation."""
    message = {
        "id": "event-123",
        "data": {"order_id": "123", "amount": 99.99, "customer": "Alice"},
        "datacontenttype": "application/json",
    }

    validated, metadata = parse_cloudevent(message, model=OrderCreated)

    assert isinstance(validated, OrderCreated)
    assert validated.order_id == "123"
    assert validated.amount == 99.99
    assert metadata["id"] == "event-123"


def test_parse_cloudevent_with_dataclass_model():
    """Test parsing CloudEvent with dataclass model."""
    message = {
        "id": "event-456",
        "data": {"shipment_id": "S123", "order_id": "O456", "carrier": "FedEx"},
    }

    validated, metadata = parse_cloudevent(message, model=ShipmentCreated)

    assert isinstance(validated, ShipmentCreated)
    assert validated.shipment_id == "S123"
    assert validated.carrier == "FedEx"


def test_parse_cloudevent_with_dict_model():
    """Test parsing CloudEvent with dict model (no validation)."""
    message = {
        "id": "event-789",
        "data": {"arbitrary": "data", "number": 42},
    }

    validated, metadata = parse_cloudevent(message, model=dict)

    assert validated == {"arbitrary": "data", "number": 42}
    assert isinstance(validated, dict)


def test_parse_cloudevent_without_model():
    """Test that parsing without model raises ValueError."""
    message = {"id": "event-123", "data": {"key": "value"}}

    with pytest.raises(ValueError, match="No model provided"):
        parse_cloudevent(message, model=None)


def test_parse_cloudevent_validation_failure():
    """Test that validation failures are properly raised."""
    message = {
        "id": "event-123",
        "data": {"order_id": "123"},  # Missing required fields
    }

    with pytest.raises(ValueError, match="Invalid CloudEvent"):
        parse_cloudevent(message, model=OrderCreated)


def test_parse_cloudevent_from_bytes():
    """Test parsing CloudEvent from bytes payload."""
    import json

    payload = json.dumps(
        {"order_id": "123", "amount": 99.99, "customer": "Bob"}
    ).encode("utf-8")

    validated, metadata = parse_cloudevent(payload, model=OrderCreated)

    assert isinstance(validated, OrderCreated)
    assert validated.order_id == "123"
    assert validated.customer == "Bob"


# Integration tests


def test_message_router_end_to_end():
    """Test complete flow from decoration to execution with validation."""

    results = []

    @message_router(pubsub="messagepubsub", topic="orders.created")
    def handle_order(message: OrderCreated):
        results.append(message)
        return "success"

    # Verify decoration
    assert hasattr(handle_order, "_is_message_handler")
    assert handle_order._is_message_handler is True

    # Simulate execution
    test_order = OrderCreated(order_id="999", amount=199.99, customer="Charlie")
    result = handle_order(test_order)

    assert result == "success"
    assert len(results) == 1
    assert results[0].order_id == "999"


def test_message_router_multiple_handlers():
    """Test multiple handlers can be decorated independently."""

    @message_router(pubsub="messagepubsub", topic="orders.created")
    def handle_order_created(message: OrderCreated):
        return "order_created"

    @message_router(pubsub="messagepubsub", topic="orders.cancelled")
    def handle_order_cancelled(message: OrderCancelled):
        return "order_cancelled"

    # Both should have independent metadata
    assert handle_order_created._message_router_data["topic"] == "orders.created"
    assert handle_order_cancelled._message_router_data["topic"] == "orders.cancelled"
    assert (
        handle_order_created._message_router_data["message_schemas"][0] == OrderCreated
    )
    assert (
        handle_order_cancelled._message_router_data["message_schemas"][0]
        == OrderCancelled
    )


def test_message_router_with_class_method():
    """Test message_router can be used with class methods."""

    class OrderHandler:
        def __init__(self):
            self.processed = []

        @message_router(pubsub="messagepubsub", topic="orders")
        def handle(self, message: OrderCreated):
            self.processed.append(message.order_id)
            return "processed"

    handler = OrderHandler()
    test_order = OrderCreated(order_id="888", amount=88.88, customer="Diana")

    result = handler.handle(test_order)

    assert result == "processed"
    assert "888" in handler.processed
    assert hasattr(handler.handle, "_is_message_handler")


# Tests for register_message_handlers


def test_register_message_handlers_discovers_standalone_function():
    """Test that standalone decorated functions are discovered."""
    mock_client = create_mock_dapr_client(["messagepubsub"])

    @message_router(
        pubsub="messagepubsub", topic="orders", dead_letter_topic="orders_DEAD"
    )
    def handle_order(message: OrderCreated):
        return "success"

    loop = asyncio.new_event_loop()
    try:
        with patch(_PATCH_TARGET, return_value=mock_client):
            closers = register_message_routes(
                dapr_client=mock_client, targets=[handle_order], loop=loop
            )
    finally:
        loop.close()

    # Should create one subscription
    assert mock_client.subscribe.call_count == 1
    assert len(closers) == 1

    # Verify subscription parameters
    call_args = mock_client.subscribe.call_args
    assert call_args.kwargs["pubsub_name"] == "messagepubsub"
    assert call_args.kwargs["topic"] == "orders"
    assert call_args.kwargs["dead_letter_topic"] == "orders_DEAD"


def test_register_message_handlers_discovers_class_methods():
    """Test that decorated methods in class instances are discovered."""
    mock_client = create_mock_dapr_client(["messagepubsub"])

    class OrderHandler:
        @message_router(pubsub="messagepubsub", topic="orders.created")
        def handle_created(self, message: OrderCreated):
            return "created"

        @message_router(pubsub="messagepubsub", topic="orders.cancelled")
        def handle_cancelled(self, message: OrderCancelled):
            return "cancelled"

    handler = OrderHandler()
    loop = asyncio.new_event_loop()
    try:
        with patch(_PATCH_TARGET, return_value=mock_client):
            closers = register_message_routes(
                dapr_client=mock_client, targets=[handler], loop=loop
            )
    finally:
        loop.close()

    # Should create two subscriptions
    assert mock_client.subscribe.call_count == 2
    assert len(closers) == 2

    # Verify both topics were registered
    topics = [call.kwargs["topic"] for call in mock_client.subscribe.call_args_list]
    assert "orders.created" in topics
    assert "orders.cancelled" in topics


def test_register_message_handlers_groups_by_topic():
    """Test that handlers sharing the same (pubsub, topic) create a single subscription."""
    mock_client = create_mock_dapr_client(["messagepubsub"])

    class OrderHandler:
        @message_router(pubsub="messagepubsub", topic="orders.events")
        def handle_created(self, message: OrderCreated):
            return "created"

        @message_router(pubsub="messagepubsub", topic="orders.events")
        def handle_cancelled(self, message: OrderCancelled):
            return "cancelled"

    handler = OrderHandler()
    loop = asyncio.new_event_loop()
    try:
        with patch(_PATCH_TARGET, return_value=mock_client):
            closers = register_message_routes(
                dapr_client=mock_client, targets=[handler], loop=loop
            )
    finally:
        loop.close()

    # Should create only one subscription (grouped by pubsub+topic)
    assert mock_client.subscribe.call_count == 1
    assert len(closers) == 1

    # Verify the subscription was created for the shared topic
    call_args = mock_client.subscribe.call_args
    assert call_args.kwargs["pubsub_name"] == "messagepubsub"
    assert call_args.kwargs["topic"] == "orders.events"

    # Both handlers are still registered and will be reachable via schema routing
    # within the composite handler created for this topic


def test_register_message_handlers_ignores_undecorated_methods():
    """Test that methods without @message_router are ignored."""
    mock_client = create_mock_dapr_client(["messagepubsub"])

    class MixedHandler:
        @message_router(pubsub="messagepubsub", topic="orders")
        def decorated_handler(self, message: OrderCreated):
            return "success"

        def regular_method(self, message: OrderCreated):
            """Not decorated, should be ignored."""
            return "ignored"

    handler = MixedHandler()
    loop = asyncio.new_event_loop()
    try:
        with patch(_PATCH_TARGET, return_value=mock_client):
            closers = register_message_routes(
                dapr_client=mock_client, targets=[handler], loop=loop
            )
    finally:
        loop.close()

    # Should only create one subscription (for decorated method)
    assert mock_client.subscribe.call_count == 1
    assert len(closers) == 1


def test_register_message_handlers_handles_multiple_targets():
    """Test registering multiple targets (functions and instances)."""
    mock_client = create_mock_dapr_client(["messagepubsub"])

    @message_router(pubsub="messagepubsub", topic="orders")
    def standalone_handler(message: OrderCreated):
        pass

    class OrderHandler:
        @message_router(pubsub="messagepubsub", topic="shipments")
        def handle_shipment(self, message: ShipmentCreated):
            pass

    handler_instance = OrderHandler()
    loop = asyncio.new_event_loop()
    try:
        with patch(_PATCH_TARGET, return_value=mock_client):
            closers = register_message_routes(
                dapr_client=mock_client,
                targets=[standalone_handler, handler_instance],
                loop=loop,
            )
    finally:
        loop.close()

    # Should create two subscriptions
    assert mock_client.subscribe.call_count == 2
    assert len(closers) == 2


def test_register_message_handlers_returns_closers():
    """Test that closer functions are returned for each subscription."""
    mock_client = create_mock_dapr_client(["messagepubsub"])

    @message_router(pubsub="messagepubsub", topic="orders.created")
    def handle_created(message: OrderCreated):
        pass

    @message_router(pubsub="messagepubsub", topic="orders.cancelled")
    def handle_cancelled(message: OrderCancelled):
        pass

    loop = asyncio.new_event_loop()
    try:
        with patch(_PATCH_TARGET, return_value=mock_client):
            closers = register_message_routes(
                dapr_client=mock_client,
                targets=[handle_created, handle_cancelled],
                loop=loop,
            )
    finally:
        loop.close()

    # Should return two closers
    assert len(closers) == 2
    assert all(callable(closer) for closer in closers)


class TestTTLDedupeBackend:
    def test_unseen_key_returns_false(self) -> None:
        backend = TTLDedupeBackend(maxsize=8, ttl=1.0)
        assert not backend.seen("event-1")

    def test_marked_key_is_seen(self) -> None:
        backend = TTLDedupeBackend(maxsize=8, ttl=1.0)
        backend.mark("event-1")
        assert backend.seen("event-1")

    def test_different_keys_are_independent(self) -> None:
        backend = TTLDedupeBackend(maxsize=8, ttl=1.0)
        backend.mark("event-1")
        assert not backend.seen("event-2")

    def test_key_expires_after_ttl(self) -> None:
        backend = TTLDedupeBackend(maxsize=8, ttl=0.1)
        backend.mark("event-1")
        assert backend.seen("event-1")
        time.sleep(0.15)
        assert not backend.seen("event-1")


# ============================================================================
# Fixtures/helpers for tests exercising the handler path
# ============================================================================


def _make_cloudevent(data: dict, **fields) -> dict:
    """Build a dict that `extract_cloudevent_data` will treat as a CloudEvent."""
    base = {
        "id": "evt-1",
        "data": data,
        "datacontenttype": "application/json",
        "pubsubname": "messagepubsub",
        "source": "/test",
        "specversion": "1.0",
        "topic": "orders",
        "type": "OrderCreated",
        "extensions": {},
    }
    base.update(fields)
    return base


def _run_messages(
    mock_dapr, mock_wf, target, messages: list[dict], *, deduper=None
) -> None:
    """Feed `messages` through the subscription thread and join cleanly."""
    mock_dapr.subscribe.return_value.__iter__.return_value = iter(messages)
    with patch(_PATCH_TARGET, return_value=mock_dapr):
        closers = register_message_routes(
            dapr_client=mock_dapr,
            targets=[target],
            wf_client=mock_wf,
            deduper=deduper,
        )
    for closer in closers:
        closer()


def _run_one_message(mock_dapr, mock_wf, target, message: dict) -> None:
    """Feed exactly one message through the subscription thread and join cleanly."""
    _run_messages(mock_dapr, mock_wf, target, messages=[message])


class _FakeTopicEventResponse:
    """Test stand-in for the SDK's TopicEventResponse.

    `conftest.py` mocks `dapr.clients.grpc._response`, so the real
    `TopicEventResponse` is a `MagicMock` inside `subscription.py` and its
    `.status` would be a chained mock. We patch it to a plain class so
    `_normalize_status` can read the string back out.
    """

    def __init__(self, status: str) -> None:
        self.status = status


@pytest.fixture
def filter_env(monkeypatch):
    """Mock dapr_client + wf_client suitable for exercising the handler path."""
    monkeypatch.setattr(
        "dapr_agents.workflow.utils.subscription.TopicEventResponse",
        _FakeTopicEventResponse,
    )
    mock_dapr = create_mock_dapr_client(["messagepubsub"])
    mock_wf = MagicMock()
    mock_wf.schedule_new_workflow.return_value = "instance-1"
    return mock_dapr, mock_wf


# ============================================================================
# Filter behavior — payload_filter / model_filter
# ============================================================================


def test_message_router_with_payload_filter():
    """payload_filter callable is stored on the decorator metadata."""

    def pf(payload, msg_ctx):
        return True

    @message_router(pubsub="messagepubsub", topic="orders", payload_filter=pf)
    def handler(message: OrderCreated):
        pass

    assert handler._message_router_data["payload_filter"] is pf
    assert handler._message_router_data["model_filter"] is None


def test_message_router_with_model_filter():
    """model_filter callable is stored on the decorator metadata."""

    def mf(model, msg_ctx):
        return True

    @message_router(pubsub="messagepubsub", topic="orders", model_filter=mf)
    def handler(message: OrderCreated):
        pass

    assert handler._message_router_data["model_filter"] is mf
    assert handler._message_router_data["payload_filter"] is None


def test_message_router_rejects_async_payload_filter():
    """Async payload_filter is rejected at decoration time."""

    async def pf(payload, msg_ctx):
        return True

    with pytest.raises(TypeError, match="payload_filter.*synchronous"):

        @message_router(pubsub="messagepubsub", topic="orders", payload_filter=pf)
        def handler(message: OrderCreated):
            pass


def test_message_router_rejects_async_model_filter():
    """Async model_filter is rejected at decoration time."""

    async def mf(model, msg_ctx):
        return True

    with pytest.raises(TypeError, match="model_filter.*synchronous"):

        @message_router(pubsub="messagepubsub", topic="orders", model_filter=mf)
        def handler(message: OrderCreated):
            pass


def test_message_router_rejects_non_callable_filter():
    """Non-callable filter values are rejected at decoration time."""

    with pytest.raises(TypeError, match="payload_filter.*callable"):

        @message_router(pubsub="messagepubsub", topic="orders", payload_filter=42)
        def handler(message: OrderCreated):
            pass


def test_message_router_rejects_callable_filter_with_async_dunder_call():
    """Callable objects whose `__call__` is `async def` are rejected too."""

    class AsyncCallableFilter:
        async def __call__(self, payload, msg_ctx):
            return True

    with pytest.raises(TypeError, match="payload_filter.*synchronous"):

        @message_router(
            pubsub="messagepubsub",
            topic="orders",
            payload_filter=AsyncCallableFilter(),
        )
        def handler(message: OrderCreated):
            pass


def test_collect_bindings_carries_decorator_filters():
    """_collect_message_bindings propagates filters from decorator metadata."""
    from dapr_agents.workflow.utils.registration import _collect_message_bindings

    def pf(payload, msg_ctx):
        return True

    def mf(model, msg_ctx):
        return True

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        payload_filter=pf,
        model_filter=mf,
    )
    def handler(message: OrderCreated):
        pass

    bindings = _collect_message_bindings(targets=[handler], routes=None)
    assert len(bindings) == 1
    assert bindings[0].payload_filter is pf
    assert bindings[0].model_filter is mf


def test_pubsub_route_spec_filter_wins_over_decorator():
    """When PubSubRouteSpec carries a filter, it overrides the decorator's."""
    from dapr_agents.types.workflow import PubSubRouteSpec
    from dapr_agents.workflow.utils.registration import _collect_message_bindings

    def deco_pf(payload, msg_ctx):
        return True

    def spec_pf(payload, msg_ctx):
        return False

    @message_router(pubsub="messagepubsub", topic="orders", payload_filter=deco_pf)
    def handler(message: OrderCreated):
        pass

    spec = PubSubRouteSpec(
        pubsub_name="messagepubsub",
        topic="orders",
        handler_fn=handler,
        payload_filter=spec_pf,
    )
    bindings = _collect_message_bindings(targets=None, routes=[spec])
    assert bindings[0].payload_filter is spec_pf


def test_pubsub_route_spec_rejects_async_filter():
    """Async filter via PubSubRouteSpec is rejected at binding collection."""
    from dapr_agents.types.workflow import PubSubRouteSpec
    from dapr_agents.workflow.utils.registration import _collect_message_bindings

    async def pf(payload, msg_ctx):
        return True

    def handler(message: OrderCreated):
        pass

    spec = PubSubRouteSpec(
        pubsub_name="messagepubsub",
        topic="orders",
        handler_fn=handler,
        payload_filter=pf,
    )
    with pytest.raises(TypeError, match="payload_filter.*synchronous"):
        _collect_message_bindings(targets=None, routes=[spec])


def test_pubsub_route_spec_rejects_non_callable_filter():
    """Non-callable filter via PubSubRouteSpec is rejected at binding collection."""
    from dapr_agents.types.workflow import PubSubRouteSpec
    from dapr_agents.workflow.utils.registration import _collect_message_bindings

    def handler(message: OrderCreated):
        pass

    spec = PubSubRouteSpec(
        pubsub_name="messagepubsub",
        topic="orders",
        handler_fn=handler,
        model_filter=42,
    )
    with pytest.raises(TypeError, match="model_filter.*callable"):
        _collect_message_bindings(targets=None, routes=[spec])


def test_pubsub_route_spec_falls_back_to_decorator_filter():
    """When PubSubRouteSpec omits a filter, the decorator's is used."""
    from dapr_agents.types.workflow import PubSubRouteSpec
    from dapr_agents.workflow.utils.registration import _collect_message_bindings

    def deco_mf(model, msg_ctx):
        return True

    @message_router(pubsub="messagepubsub", topic="orders", model_filter=deco_mf)
    def handler(message: OrderCreated):
        pass

    spec = PubSubRouteSpec(
        pubsub_name="messagepubsub", topic="orders", handler_fn=handler
    )
    bindings = _collect_message_bindings(targets=None, routes=[spec])
    assert bindings[0].model_filter is deco_mf


# Behavioral tests via mock subscription


def test_payload_filter_accept_schedules_workflow(filter_env):
    mock_dapr, mock_wf = filter_env

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        payload_filter=lambda p, msg_ctx: p.get("amount", 0) > 50,
    )
    def handler(message: OrderCreated):
        return "ok"

    msg = _make_cloudevent({"order_id": "1", "amount": 100.0, "customer": "Alice"})
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert mock_wf.schedule_new_workflow.called


def test_payload_filter_reject_skips_workflow(filter_env):
    mock_dapr, mock_wf = filter_env

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        payload_filter=lambda p, msg_ctx: p.get("amount", 0) > 50,
    )
    def handler(message: OrderCreated):
        return "ok"

    msg = _make_cloudevent({"order_id": "1", "amount": 10.0, "customer": "Alice"})
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert not mock_wf.schedule_new_workflow.called


def test_model_filter_reject_skips_workflow(filter_env):
    mock_dapr, mock_wf = filter_env

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        model_filter=lambda m, msg_ctx: m.amount > 50,
    )
    def handler(message: OrderCreated):
        return "ok"

    msg = _make_cloudevent({"order_id": "1", "amount": 10.0, "customer": "Alice"})
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert not mock_wf.schedule_new_workflow.called


def test_filter_exception_skips_binding(filter_env):
    mock_dapr, mock_wf = filter_env

    def buggy(model, msg_ctx):
        raise RuntimeError("boom")

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        model_filter=buggy,
    )
    def handler(message: OrderCreated):
        return "ok"

    msg = _make_cloudevent({"order_id": "1", "amount": 100.0, "customer": "Alice"})
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert not mock_wf.schedule_new_workflow.called


def test_model_filter_rejection_skips_remaining_schemas_of_binding(filter_env):
    """A model_filter rejection on one schema must skip the binding's other schemas."""
    mock_dapr, mock_wf = filter_env
    call_count = {"n": 0}

    def rejecting_model_filter(model, msg_ctx):
        call_count["n"] += 1
        return False

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=Union[OrderCreated, OrderCancelled],
        model_filter=rejecting_model_filter,
    )
    def handler(message):
        return "ok"

    # Loose payload that validates against both OrderCreated and OrderCancelled
    msg = _make_cloudevent(
        {"order_id": "1", "amount": 100.0, "customer": "Alice"}, type="Unknown"
    )
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert call_count["n"] == 1
    assert not mock_wf.schedule_new_workflow.called


def test_payload_filter_runs_once_per_binding_for_union(filter_env):
    """For a binding with multiple schemas, payload_filter is evaluated once."""
    mock_dapr, mock_wf = filter_env
    call_count = {"n": 0}

    def counting_filter(payload, msg_ctx):
        call_count["n"] += 1
        return False  # reject so we exhaust every schema slot

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=Union[OrderCreated, OrderCancelled],
        payload_filter=counting_filter,
    )
    def handler(message):
        return "ok"

    msg = _make_cloudevent(
        {"order_id": "1", "amount": 100.0, "customer": "Alice"}, type="Unknown"
    )
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert call_count["n"] == 1
    assert not mock_wf.schedule_new_workflow.called


def test_filter_message_context_exposes_event_and_handler_name(filter_env):
    mock_dapr, mock_wf = filter_env
    captured: list = []

    def capturing(model, msg_ctx):
        captured.append(msg_ctx)
        return True

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        model_filter=capturing,
    )
    def handler(message: OrderCreated):
        return "ok"

    msg = _make_cloudevent(
        {"order_id": "1", "amount": 100.0, "customer": "Alice"},
        id="evt-42",
        source="/api/orders",
        type="OrderCreated",
    )
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert len(captured) == 1
    msg_ctx = captured[0]
    assert msg_ctx.event.id == "evt-42"
    assert msg_ctx.event.source == "/api/orders"
    assert msg_ctx.event.topic == "orders"
    assert msg_ctx.event.type == "OrderCreated"
    assert msg_ctx.handler_name == "handler"


# ============================================================================
# Mapping behavior
# ============================================================================


def test_message_router_with_mapper():
    """mapper callable is stored on the decorator metadata."""

    def mapper(model, msg_ctx):
        return model

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=OrderCreated,
        mapper=mapper,
    )
    def handler(message):
        pass

    assert handler._message_router_data["mapper"] is mapper


def test_message_router_rejects_async_mapper():
    """Async mapper is rejected at decoration time."""

    async def mapper(model, msg_ctx):
        return True

    with pytest.raises(TypeError, match="mapper.*synchronous"):

        @message_router(
            pubsub="messagepubsub",
            topic="orders",
            message_model=OrderCreated,
            mapper=mapper,
        )
        def handler(message):
            pass


def test_message_router_rejects_non_callable_mapper():
    """Non-callable mapper values are rejected at decoration time."""

    with pytest.raises(TypeError, match="mapper.*callable"):

        @message_router(
            pubsub="messagepubsub",
            topic="orders",
            message_model=OrderCreated,
            mapper=False,
        )
        def handler(message):
            pass


def test_message_router_rejects_callable_mapper_with_async_dunder_call():
    """Callable mapper objects whose `__call__` is `async def` are rejected too."""

    class AsyncCallableMapper:
        async def __call__(self, model, msg_ctx):
            return {}

    with pytest.raises(TypeError, match="mapper.*synchronous"):

        @message_router(
            pubsub="messagepubsub",
            topic="orders",
            message_model=OrderCreated,
            mapper=AsyncCallableMapper(),
        )
        def handler(message):
            pass


def test_collect_bindings_carries_decorator_mapper():
    """_collect_message_bindings propagates mapper from decorator metadata."""
    from dapr_agents.workflow.utils.registration import _collect_message_bindings

    def mapper(payload, msg_ctx):
        return {}

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=OrderCreated,
        mapper=mapper,
    )
    def handler(message):
        pass

    bindings = _collect_message_bindings(targets=[handler], routes=None)
    assert len(bindings) == 1
    assert bindings[0].mapper is mapper


def test_pubsub_route_spec_mapper_wins_over_decorator():
    """When PubSubRouteSpec carries a mapper, it overrides the decorator's."""
    from dapr_agents.types.workflow import PubSubRouteSpec
    from dapr_agents.workflow.utils.registration import _collect_message_bindings

    def deco_mapper(payload, msg_ctx):
        return {}

    def spec_mapper(payload, msg_ctx):
        return {}

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=OrderCreated,
        mapper=deco_mapper,
    )
    def handler(message):
        pass

    spec = PubSubRouteSpec(
        pubsub_name="messagepubsub",
        topic="orders",
        handler_fn=handler,
        message_model=OrderCreated,
        mapper=spec_mapper,
    )
    bindings = _collect_message_bindings(targets=None, routes=[spec])
    assert bindings[0].mapper is spec_mapper


def test_pubsub_route_spec_rejects_async_mapper():
    """Async mapper via PubSubRouteSpec is rejected at binding collection."""
    from dapr_agents.types.workflow import PubSubRouteSpec
    from dapr_agents.workflow.utils.registration import _collect_message_bindings

    async def mapper(payload, msg_ctx):
        return {}

    def handler(message):
        pass

    spec = PubSubRouteSpec(
        pubsub_name="messagepubsub",
        topic="orders",
        handler_fn=handler,
        message_model=OrderCreated,
        mapper=mapper,
    )
    with pytest.raises(TypeError, match="mapper.*synchronous"):
        _collect_message_bindings(targets=None, routes=[spec])


def test_pubsub_route_spec_rejects_non_callable_mapper():
    """Non-callable mapper via PubSubRouteSpec is rejected at binding collection."""
    from dapr_agents.types.workflow import PubSubRouteSpec
    from dapr_agents.workflow.utils.registration import _collect_message_bindings

    def handler(message):
        pass

    spec = PubSubRouteSpec(
        pubsub_name="messagepubsub",
        topic="orders",
        handler_fn=handler,
        message_model=OrderCreated,
        mapper=False,
    )
    with pytest.raises(TypeError, match="mapper.*callable"):
        _collect_message_bindings(targets=None, routes=[spec])


def test_pubsub_route_spec_falls_back_to_decorator_mapper():
    """When PubSubRouteSpec omits a mapper, the decorator's is used."""
    from dapr_agents.types.workflow import PubSubRouteSpec
    from dapr_agents.workflow.utils.registration import _collect_message_bindings

    def deco_mapper(payload, msg_ctx):
        return {}

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=OrderCreated,
        mapper=deco_mapper,
    )
    def handler(message):
        pass

    spec = PubSubRouteSpec(
        pubsub_name="messagepubsub", topic="orders", handler_fn=handler
    )
    bindings = _collect_message_bindings(targets=None, routes=[spec])
    assert bindings[0].mapper is deco_mapper


# Behavioral tests via mock subscription


def test_mapper_dict_model_schedules_workflow(filter_env):
    mock_dapr, mock_wf = filter_env

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=OrderCreated,
        mapper=lambda m, msg_ctx: {"foo": "bar"},
    )
    def handler(message):
        return "ok"

    msg = _make_cloudevent({"order_id": "1", "amount": 1000.0, "customer": "Bob"})
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert mock_wf.schedule_new_workflow.called
    assert mock_wf.schedule_new_workflow.call_args.kwargs["input"]["foo"] == "bar"


def test_mapper_pydantic_model_schedules_workflow(filter_env):
    mock_dapr, mock_wf = filter_env

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=OrderCreated,
        mapper=lambda m, msg_ctx: OrderCancelled(
            order_id=m.order_id, reason="Buyers remorse"
        ),
    )
    def handler(message):
        return "ok"

    msg = _make_cloudevent({"order_id": "1", "amount": 1000.0, "customer": "Bob"})
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert mock_wf.schedule_new_workflow.called
    assert mock_wf.schedule_new_workflow.call_args.kwargs["input"]["order_id"] == "1"
    assert (
        mock_wf.schedule_new_workflow.call_args.kwargs["input"]["reason"]
        == "Buyers remorse"
    )


def test_mapper_dataclass_model_schedules_workflow(filter_env):
    mock_dapr, mock_wf = filter_env

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=OrderCreated,
        mapper=lambda m, msg_ctx: ShipmentCreated(
            shipment_id="s1", order_id=m.order_id, carrier="UPS"
        ),
    )
    def handler(message):
        return "ok"

    msg = _make_cloudevent({"order_id": "1", "amount": 1000.0, "customer": "Bob"})
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert mock_wf.schedule_new_workflow.called
    assert (
        mock_wf.schedule_new_workflow.call_args.kwargs["input"]["shipment_id"] == "s1"
    )
    assert mock_wf.schedule_new_workflow.call_args.kwargs["input"]["order_id"] == "1"
    assert mock_wf.schedule_new_workflow.call_args.kwargs["input"]["carrier"] == "UPS"


def test_mapper_mutated_model_schedules_workflow(filter_env):
    mock_dapr, mock_wf = filter_env

    def mapper(model: OrderCreated, msg_ctx):
        model.amount = 0.0
        return model

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=OrderCreated,
        mapper=mapper,
    )
    def handler(message):
        return "ok"

    msg = _make_cloudevent({"order_id": "1", "amount": 1000.0, "customer": "Bob"})
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert mock_wf.schedule_new_workflow.called
    assert mock_wf.schedule_new_workflow.call_args.kwargs["input"]["order_id"] == "1"
    assert mock_wf.schedule_new_workflow.call_args.kwargs["input"]["amount"] == 0.0
    assert mock_wf.schedule_new_workflow.call_args.kwargs["input"]["customer"] == "Bob"


def test_mapper_non_json_serializable_model_skips_binding(filter_env):
    mock_dapr, mock_wf = filter_env

    def mapper(value, msg_ctx):
        return False

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=OrderCreated,
        mapper=mapper,
    )
    def handler(message):
        return "ok"

    msg = _make_cloudevent({"order_id": "1", "amount": 1000.0, "customer": "Bob"})
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert not mock_wf.schedule_new_workflow.called


def test_mapper_exception_skips_binding(filter_env):
    mock_dapr, mock_wf = filter_env

    def mapper(model, msg_ctx):
        raise RuntimeError("fahh")

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=OrderCreated,
        mapper=mapper,
    )
    def handler(message):
        return "ok"

    msg = _make_cloudevent({"order_id": "1", "amount": 1000.0, "customer": "Bob"})
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert not mock_wf.schedule_new_workflow.called


def test_mapper_failure_skips_remaining_schemas_of_binding(filter_env):
    """A mapper failure on one schema must skip the binding's other schemas."""
    mock_dapr, mock_wf = filter_env
    call_count = {"n": 0}

    def failing_mapper(model, msg_ctx):
        call_count["n"] += 1
        return False

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=Union[OrderCreated, OrderCancelled],
        mapper=failing_mapper,
    )
    def handler(message):
        return "ok"

    # Loose payload that validates against both OrderCreated and OrderCancelled
    msg = _make_cloudevent(
        {"order_id": "1", "amount": 1000.0, "customer": "Bob"}, type="Unknown"
    )
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert call_count["n"] == 1
    assert not mock_wf.schedule_new_workflow.called


def test_mapper_runs_once_per_binding_for_union(filter_env):
    """For a binding with multiple schemas, mapper is evaluated once."""
    mock_dapr, mock_wf = filter_env
    call_count = {"n": 0}

    def counting_mapper(model, msg_ctx):
        call_count["n"] += 1
        return False  # mapper always fails so we exhaust every schema slot

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=Union[OrderCreated, OrderCancelled],
        mapper=counting_mapper,
    )
    def handler(message):
        return "ok"

    msg = _make_cloudevent(
        {"order_id": "1", "amount": 1000.0, "customer": "Bob"}, type="Unknown"
    )
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert call_count["n"] == 1
    assert not mock_wf.schedule_new_workflow.called


def test_mapper_message_context_exposes_event_and_handler_name(filter_env):
    mock_dapr, mock_wf = filter_env
    captured: list = []

    def capturing(model, msg_ctx):
        captured.append(msg_ctx)
        return model

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        message_model=OrderCreated,
        mapper=capturing,
    )
    def handler(message):
        return "ok"

    msg = _make_cloudevent(
        {"order_id": "1", "amount": 1000.0, "customer": "Bob"},
        id="evt-67",
        source="/api/orders",
        type="OrderCreated",
    )
    _run_one_message(mock_dapr, mock_wf, handler, msg)

    assert len(captured) == 1
    msg_ctx = captured[0]
    assert msg_ctx.event.id == "evt-67"
    assert msg_ctx.event.source == "/api/orders"
    assert msg_ctx.event.topic == "orders"
    assert msg_ctx.event.type == "OrderCreated"
    assert msg_ctx.handler_name == "handler"


# ============================================================================
# Dedup behavior — mark on terminal outcome, not on arrival
# ============================================================================


def test_dedup_success_drops_redelivery(filter_env):
    """SUCCESS marks the dedup cache so a redelivery of the same id is ack'd silently."""
    mock_dapr, mock_wf = filter_env

    @message_router(pubsub="messagepubsub", topic="orders")
    def handler(message: OrderCreated):
        return "ok"

    deduper = TTLDedupeBackend(maxsize=8, ttl=60.0)
    msg = _make_cloudevent(
        {"order_id": "1", "amount": 100.0, "customer": "Alice"}, id="evt-1"
    )
    _run_messages(mock_dapr, mock_wf, handler, [msg, msg], deduper=deduper)

    assert mock_wf.schedule_new_workflow.call_count == 1
    assert deduper.seen("evt-1")


def test_dedup_drop_also_marks(filter_env):
    """DROP (filter rejection) is terminal too — the duplicate must be dropped silently."""
    mock_dapr, mock_wf = filter_env

    @message_router(
        pubsub="messagepubsub",
        topic="orders",
        model_filter=lambda m, msg_ctx: False,  # always reject -> DROP
    )
    def handler(message: OrderCreated):
        return "ok"

    deduper = TTLDedupeBackend(maxsize=8, ttl=60.0)
    msg = _make_cloudevent(
        {"order_id": "1", "amount": 100.0, "customer": "Alice"}, id="evt-1"
    )
    _run_messages(mock_dapr, mock_wf, handler, [msg, msg], deduper=deduper)

    assert mock_wf.schedule_new_workflow.call_count == 0
    assert deduper.seen("evt-1")


def test_dedup_retry_does_not_mark(filter_env):
    """RETRY must NOT mark; otherwise broker redelivery is silently neutralized."""
    mock_dapr, mock_wf = filter_env
    mock_wf.schedule_new_workflow.side_effect = RuntimeError("transient")

    @message_router(pubsub="messagepubsub", topic="orders")
    def handler(message: OrderCreated):
        return "ok"

    deduper = TTLDedupeBackend(maxsize=8, ttl=60.0)
    msg = _make_cloudevent(
        {"order_id": "1", "amount": 100.0, "customer": "Alice"}, id="evt-1"
    )
    _run_messages(mock_dapr, mock_wf, handler, [msg, msg], deduper=deduper)

    # Both deliveries reached the handler (RETRY left the cache empty).
    assert mock_wf.schedule_new_workflow.call_count == 2
    assert not deduper.seen("evt-1")


def test_dedup_disabled_processes_every_message(filter_env):
    """No deduper = every delivery processes, even with identical ids."""
    mock_dapr, mock_wf = filter_env

    @message_router(pubsub="messagepubsub", topic="orders")
    def handler(message: OrderCreated):
        return "ok"

    msg = _make_cloudevent(
        {"order_id": "1", "amount": 100.0, "customer": "Alice"}, id="evt-1"
    )
    _run_messages(mock_dapr, mock_wf, handler, [msg, msg], deduper=None)

    assert mock_wf.schedule_new_workflow.call_count == 2


def _binding(name, *, schemas, payload_filter=None, model_filter=None):
    return MessageRouteBinding(
        handler=lambda: None,
        schemas=schemas,
        pubsub="messagepubsub",
        topic="orders",
        dead_letter_topic=None,
        name=name,
        payload_filter=payload_filter,
        model_filter=model_filter,
    )


def test_warn_unreachable_when_unconditional_binding_registered_first(caplog):
    bindings = [
        _binding("first", schemas=[OrderCreated]),
        _binding("second", schemas=[OrderCreated]),
    ]
    with caplog.at_level(logging.WARNING):
        _warn_unreachable_bindings(bindings)

    assert "'second' never runs" in caplog.text
    assert "'first'" in caplog.text


def test_no_warn_when_first_binding_has_filter(caplog):
    bindings = [
        _binding("specific", schemas=[OrderCreated], model_filter=lambda m, c: True),
        _binding("fallback", schemas=[OrderCreated]),
    ]
    with caplog.at_level(logging.WARNING):
        _warn_unreachable_bindings(bindings)

    assert caplog.text == ""


def test_no_warn_when_schemas_differ(caplog):
    bindings = [
        _binding("a", schemas=[OrderCreated]),
        _binding("b", schemas=[OrderCancelled]),
    ]
    with caplog.at_level(logging.WARNING):
        _warn_unreachable_bindings(bindings)

    assert caplog.text == ""


# ============================================================================
# Helper unit tests — subscription.py module-level functions
# ============================================================================


def _mk_binding(
    name,
    *,
    topic="orders",
    pubsub="messagepubsub",
    schemas=None,
    dead_letter_topic=None,
):
    return MessageRouteBinding(
        handler=lambda: None,
        schemas=schemas if schemas is not None else [OrderCreated],
        pubsub=pubsub,
        topic=topic,
        dead_letter_topic=dead_letter_topic,
        name=name,
    )


def _raise_runtime(*args, **kwargs):
    raise RuntimeError("no loop")


def _msg_ctx(handler_name: str = "handler") -> MessageContext:
    event = EventMessageMetadata(
        id="evt-1",
        datacontenttype="application/json",
        pubsubname="messagepubsub",
        source="/test",
        specversion="1.0",
        time=None,
        topic="orders",
        traceid=None,
        traceparent=None,
        type="OrderCreated",
        tracestate=None,
        headers={},
    )
    return MessageContext(event=event, handler_name=handler_name)


# ---- validate_hooks ------------------------------------------------------


def test_validate_hooks_none_and_sync_are_ok():
    validate_hooks(None, "payload_filter")
    validate_hooks(lambda v, c: True, "model_filter")
    validate_hooks(lambda v, c: False, "mapper")


def test_validate_hooks_rejects_non_callable():
    with pytest.raises(TypeError, match="payload_filter.*callable"):
        validate_hooks(42, "payload_filter")


def test_validate_hooks_rejects_async_function():
    async def f(v, c):
        return True

    with pytest.raises(TypeError, match="model_filter.*synchronous"):
        validate_hooks(f, "model_filter")


def test_validate_hooks_rejects_async_dunder_call():
    class AsyncCallable:
        async def __call__(self, v, c):
            return True

    with pytest.raises(TypeError, match="mapper.*synchronous"):
        validate_hooks(AsyncCallable(), "mapper")


# ---- _validate_delivery_mode ----------------------------------------------


def test_validate_delivery_mode_accepts_known_modes():
    _validate_delivery_mode(DELIVERY_MODE_SYNC)
    _validate_delivery_mode(DELIVERY_MODE_ASYNC)


def test_validate_delivery_mode_rejects_unknown():
    with pytest.raises(ValueError, match="delivery_mode must be"):
        _validate_delivery_mode("turbo")


# ---- _validate_dead_letter_topics -----------------------------------------


def test_validate_dlq_conflict_raises():
    bindings = [
        _mk_binding("a", dead_letter_topic="dlq-1"),
        _mk_binding("b", dead_letter_topic="dlq-2"),
    ]
    with pytest.raises(ValueError, match="Multiple dead_letter_topics"):
        _validate_dead_letter_topics(bindings)


def test_validate_dlq_same_value_is_ok():
    bindings = [
        _mk_binding("a", dead_letter_topic="dlq-1"),
        _mk_binding("b", dead_letter_topic="dlq-1"),
    ]
    _validate_dead_letter_topics(bindings)


def test_validate_dlq_none_does_not_conflict():
    bindings = [
        _mk_binding("a", dead_letter_topic="dlq-1"),
        _mk_binding("b", dead_letter_topic=None),
    ]
    _validate_dead_letter_topics(bindings)


def test_validate_dlq_different_topics_are_independent():
    bindings = [
        _mk_binding("a", topic="orders", dead_letter_topic="dlq-1"),
        _mk_binding("b", topic="shipments", dead_letter_topic="dlq-2"),
    ]
    _validate_dead_letter_topics(bindings)


# ---- _group_bindings_by_topic ---------------------------------------------


def test_group_bindings_by_topic_groups_shared_keys():
    a = _mk_binding("a", topic="orders")
    b = _mk_binding("b", topic="orders")
    c = _mk_binding("c", topic="shipments")

    grouped = _group_bindings_by_topic([a, b, c])

    assert set(grouped) == {
        ("messagepubsub", "orders"),
        ("messagepubsub", "shipments"),
    }
    assert grouped[("messagepubsub", "orders")] == [a, b]
    assert grouped[("messagepubsub", "shipments")] == [c]


# ---- _build_binding_schema_pairs ------------------------------------------


def test_build_binding_schema_pairs_flattens_schemas():
    binding = _mk_binding("b", schemas=[OrderCreated, OrderCancelled])
    pairs = _build_binding_schema_pairs([binding])
    assert pairs == [(binding, OrderCreated), (binding, OrderCancelled)]


def test_build_binding_schema_pairs_defaults_to_dict_when_empty():
    binding = _mk_binding("b", schemas=[])
    pairs = _build_binding_schema_pairs([binding])
    assert pairs == [(binding, dict)]


# ---- _order_pairs_by_cloudevent_type --------------------------------------


def test_order_pairs_no_type_returns_input_unchanged():
    binding = _mk_binding("b")
    pairs = [(binding, OrderCreated), (binding, OrderCancelled)]
    assert _order_pairs_by_cloudevent_type(pairs, None) == pairs


def test_order_pairs_prioritizes_matching_type():
    binding = _mk_binding("b")
    pairs = [(binding, OrderCreated), (binding, OrderCancelled)]
    ordered = _order_pairs_by_cloudevent_type(pairs, "OrderCancelled")
    assert ordered == [(binding, OrderCancelled), (binding, OrderCreated)]


def test_order_pairs_unmatched_type_returns_input_unchanged():
    binding = _mk_binding("b")
    pairs = [(binding, OrderCreated), (binding, OrderCancelled)]
    assert _order_pairs_by_cloudevent_type(pairs, "Unknown") == pairs


# ---- _normalize_status -----------------------------------------------------


def test_normalize_status_from_string():
    assert _normalize_status("SUCCESS") == STATUS_SUCCESS
    assert _normalize_status("retry") == STATUS_RETRY
    assert _normalize_status("DROP") == STATUS_DROP


def test_normalize_status_from_enum():
    assert _normalize_status(TopicEventResponseStatus.success) == STATUS_SUCCESS
    assert _normalize_status(TopicEventResponseStatus.retry) == STATUS_RETRY
    assert _normalize_status(TopicEventResponseStatus.drop) == STATUS_DROP


def test_normalize_status_unknown_returns_none():
    assert _normalize_status("teapot") is None


# ---- _filter_accepts -------------------------------------------------------


def test_filter_accepts_none_filter_returns_true():
    result = _filter_accepts(
        None, {"a": 1}, _msg_ctx(), kind="payload_filter", binding_name="b"
    )
    assert result is True


def test_filter_accepts_forwards_value_and_context():
    seen = {}

    def capture(value, ctx):
        seen["value"] = value
        seen["ctx"] = ctx
        return True

    value = {"a": 1}
    ctx = _msg_ctx()
    result = _filter_accepts(
        capture, value, ctx, kind="payload_filter", binding_name="b"
    )
    assert result is True
    assert seen["value"] is value
    assert seen["ctx"] is ctx


def test_filter_accepts_false_rejects():
    result = _filter_accepts(
        lambda v, c: False, {}, _msg_ctx(), kind="model_filter", binding_name="b"
    )
    assert result is False


def test_filter_accepts_coerces_truthy_result_to_bool():
    result = _filter_accepts(
        lambda v, c: "yes", {}, _msg_ctx(), kind="model_filter", binding_name="b"
    )
    assert result is True


def test_filter_accepts_swallows_exception_as_rejection(caplog):
    def boom(value, ctx):
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        result = _filter_accepts(
            boom, {}, _msg_ctx(), kind="payload_filter", binding_name="bind-x"
        )
    assert result is False
    assert "bind-x" in caplog.text


# ---- _safe_map ------------------------------------------


def test_safe_map_passes_through_value_when_no_mapper():
    value = {"messi": 10}
    result = _safe_map(None, value, _msg_ctx(), binding_name="pr")
    assert result is value


def test_safe_map_forwards_value_and_context():
    seen = {}

    def mapper(value, ctx):
        seen["value"] = value
        seen["ctx"] = ctx
        return {"lamont yall": 10}

    value = {"lamont yall": 10}
    ctx = _msg_ctx()
    result = _safe_map(mapper, value, ctx, binding_name="jamal")
    assert result == value
    assert seen["value"] is value
    assert seen["ctx"] is ctx


def test_safe_map_allows_mutation():
    def mapper(value, ctx):
        value["endo"] = 3
        return value

    value = {"mitoma": 9}
    result = _safe_map(mapper, value, _msg_ctx(), binding_name="jpn")
    assert result is value


def test_safe_map_returns_dict():
    def mapper(value, ctx):
        return {"mbappe": 10}

    result = _safe_map(mapper, {}, _msg_ctx(), binding_name="offside")
    assert result == {"mbappe": 10}


def test_safe_map_returns_pydantic_model():
    def mapper(value, ctx):
        return OrderCreated(
            order_id="therapy-sessions-1",
            amount=50000000.0,
            customer="italian sports fans",
        )

    result = _safe_map(mapper, {}, _msg_ctx(), binding_name="therapy")
    assert result == OrderCreated(
        order_id="therapy-sessions-1", amount=50000000.0, customer="italian sports fans"
    )


def test_safe_map_returns_dataclass():
    def mapper(value, ctx):
        return ShipmentCreated(
            shipment_id="s1",
            order_id="therapy-sessions-1",
            carrier="italian therapists",
        )

    result = _safe_map(mapper, {}, _msg_ctx(), binding_name="therapy")
    assert result == ShipmentCreated(
        shipment_id="s1", order_id="therapy-sessions-1", carrier="italian therapists"
    )


def test_safe_map_non_json_serializable_model_returns_none():
    def mapper(value, ctx):
        return "handball"

    result = _safe_map(mapper, {}, _msg_ctx(), binding_name="var")
    assert result is None


def test_safe_map_swallows_exception_and_returns_none(caplog):
    def mapper(value, ctx):
        raise RuntimeError("yellow")

    with caplog.at_level(logging.ERROR):
        result = _safe_map(mapper, {}, _msg_ctx(), binding_name="flop")
    assert result is None
    assert "flop" in caplog.text


# ---- _attach_metadata_to_payload ------------------------------------------


def test_attach_metadata_to_dict():
    payload = {"order_id": "1"}
    _attach_metadata_to_payload(payload, {"id": "evt-1"})
    assert payload[METADATA_KEY] == {"id": "evt-1"}


def test_attach_metadata_none_is_noop():
    payload = {"order_id": "1"}
    _attach_metadata_to_payload(payload, None)
    assert METADATA_KEY not in payload


def test_attach_metadata_to_plain_object():
    class Obj:
        pass

    obj = Obj()
    _attach_metadata_to_payload(obj, {"id": "evt-1"})
    assert getattr(obj, METADATA_KEY) == {"id": "evt-1"}


def test_attach_metadata_swallows_setattr_error():
    model = OrderCreated(order_id="1", amount=1.0, customer="Alice")
    _attach_metadata_to_payload(model, {"id": "evt-1"})  # must not raise


# ---- _serialize_workflow_input --------------------------------------------


def test_serialize_workflow_input_dict_without_metadata():
    wf_input, metadata = _serialize_workflow_input({"a": 1})
    assert wf_input == {"a": 1}
    assert metadata is None


def test_serialize_workflow_input_dict_with_metadata_copies():
    src = {"a": 1, METADATA_KEY: {"id": "evt-1"}}
    wf_input, metadata = _serialize_workflow_input(src)

    assert metadata == {"id": "evt-1"}
    assert wf_input[METADATA_KEY] == {"id": "evt-1"}
    # the metadata dict is copied into the workflow input, not aliased
    assert wf_input[METADATA_KEY] is not src[METADATA_KEY]
    # the source payload is not mutated
    wf_input["a"] = 999
    assert src["a"] == 1


def test_serialize_workflow_input_pydantic_model():
    model = OrderCreated(order_id="1", amount=2.0, customer="Alice")
    wf_input, metadata = _serialize_workflow_input(model)
    assert wf_input == {"order_id": "1", "amount": 2.0, "customer": "Alice"}
    assert metadata is None


def test_serialize_workflow_input_dataclass():
    shipment = ShipmentCreated(shipment_id="s1", order_id="o1", carrier="UPS")
    wf_input, metadata = _serialize_workflow_input(shipment)
    assert wf_input == {"shipment_id": "s1", "order_id": "o1", "carrier": "UPS"}
    assert metadata is None


def test_serialize_workflow_input_other_wraps_in_data():
    wf_input, metadata = _serialize_workflow_input(42)
    assert wf_input == {"data": 42}
    assert metadata is None


# ---- _resolve_event_loop ---------------------------------------------------


def test_resolve_event_loop_returns_provided_loop():
    loop = asyncio.new_event_loop()
    try:
        assert _resolve_event_loop(loop) is loop
    finally:
        loop.close()


def test_resolve_event_loop_uses_running_loop(monkeypatch):
    sentinel = asyncio.new_event_loop()
    try:
        monkeypatch.setattr(asyncio, "get_running_loop", lambda: sentinel)
        assert _resolve_event_loop(None) is sentinel
    finally:
        sentinel.close()


def test_resolve_event_loop_raises_without_any_loop(monkeypatch):
    monkeypatch.setattr(asyncio, "get_running_loop", _raise_runtime)
    monkeypatch.setattr(asyncio, "get_event_loop", _raise_runtime)
    with pytest.raises(RuntimeError, match="No running event loop available"):
        _resolve_event_loop(None)


def test_resolve_event_loop_raises_on_closed_loop(monkeypatch):
    closed = asyncio.new_event_loop()
    closed.close()
    monkeypatch.setattr(asyncio, "get_running_loop", _raise_runtime)
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: closed)
    with pytest.raises(RuntimeError, match="No running event loop available") as exc:
        _resolve_event_loop(None)
    assert "Event loop is closed" in str(exc.value.__cause__)


# ---- _cancel_tasks ---------------------------------------------------------


def test_cancel_tasks_cancels_each_task():
    task_a, task_b = MagicMock(), MagicMock()
    _cancel_tasks([task_a, task_b])
    task_a.cancel.assert_called_once()
    task_b.cancel.assert_called_once()


def test_cancel_tasks_swallows_errors_and_continues():
    failing = MagicMock()
    failing.cancel.side_effect = RuntimeError("nope")
    healthy = MagicMock()
    _cancel_tasks([failing, healthy])  # must not raise
    healthy.cancel.assert_called_once()


# ---- _shutdown_thread ------------------------------------------------------


def test_shutdown_thread_closes_subscription_and_joins():
    thread = MagicMock()
    thread.is_alive.return_value = False
    subscription = MagicMock()
    _shutdown_thread(thread, subscription, "messagepubsub", "orders")
    subscription.close.assert_called_once()
    thread.join.assert_called_once()


def test_shutdown_thread_zombie_non_daemon_raises():
    thread = MagicMock()
    thread.is_alive.return_value = True
    thread.daemon = False
    with pytest.raises(RuntimeError, match="did not stop"):
        _shutdown_thread(thread, MagicMock(), "messagepubsub", "orders")


def test_shutdown_thread_zombie_daemon_warns(caplog):
    thread = MagicMock()
    thread.is_alive.return_value = True
    thread.daemon = True
    with caplog.at_level(logging.WARNING):
        _shutdown_thread(thread, MagicMock(), "messagepubsub", "orders")
    assert "daemon thread" in caplog.text


def test_shutdown_thread_swallows_close_error():
    thread = MagicMock()
    thread.is_alive.return_value = False
    subscription = MagicMock()
    subscription.close.side_effect = RuntimeError("boom")
    _shutdown_thread(thread, subscription, "messagepubsub", "orders")  # must not raise
    thread.join.assert_called_once()


# ---- _log_workflow_outcome -------------------------------------------------


def test_log_workflow_outcome_no_state_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _log_workflow_outcome("inst-1", None, log_outcome=True)
    assert "no state" in caplog.text


def test_log_workflow_outcome_completed_logs_when_enabled(caplog):
    state = MagicMock()
    state.runtime_status = WorkflowStatus.COMPLETED
    state.serialized_output = "out"
    with caplog.at_level(logging.DEBUG):
        _log_workflow_outcome("inst-1", state, log_outcome=True)
    assert "COMPLETED" in caplog.text


def test_log_workflow_outcome_failure_logs_error(caplog):
    state = MagicMock()
    state.runtime_status = MagicMock()  # not COMPLETED
    failure = MagicMock()
    failure.error_type = "ValueError"
    failure.message = "bad input"
    failure.stack_trace = "trace"
    state.failure_details = failure
    with caplog.at_level(logging.ERROR):
        _log_workflow_outcome("inst-1", state, log_outcome=True)
    assert "FAILED" in caplog.text
    assert "ValueError" in caplog.text


def test_log_workflow_outcome_non_completed_without_failure(caplog):
    state = MagicMock()
    state.runtime_status = MagicMock()  # not COMPLETED
    state.failure_details = None
    with caplog.at_level(logging.ERROR):
        _log_workflow_outcome("inst-1", state, log_outcome=True)
    assert "finished with status" in caplog.text
