import asyncio
import pytest
from typing import Union, Optional
from dataclasses import dataclass
from unittest.mock import MagicMock
from pydantic import BaseModel, Field

from dapr_agents.workflow.decorators.routers import message_router
from dapr_agents.workflow.utils.routers import (
    extract_message_models,
    extract_cloudevent_data,
    validate_message_model,
    parse_cloudevent,
)
from dapr_agents.workflow.utils.registration import register_message_routes


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
    mock_client = MagicMock()
    mock_client.subscribe_with_handler.return_value = MagicMock()

    @message_router(pubsub="messagepubsub", topic="orders")
    def handle_order(message: OrderCreated):
        return "success"

    loop = asyncio.new_event_loop()
    try:
        closers = register_message_routes(
            dapr_client=mock_client, targets=[handle_order], loop=loop
        )
    finally:
        loop.close()

    # Should create one subscription
    assert mock_client.subscribe_with_handler.call_count == 1
    assert len(closers) == 1

    # Verify subscription parameters
    call_args = mock_client.subscribe_with_handler.call_args
    assert call_args.kwargs["pubsub_name"] == "messagepubsub"
    assert call_args.kwargs["topic"] == "orders"
    assert call_args.kwargs["dead_letter_topic"] == "orders_DEAD"


def test_register_message_handlers_discovers_class_methods():
    """Test that decorated methods in class instances are discovered."""
    mock_client = MagicMock()
    mock_client.subscribe_with_handler.return_value = MagicMock()

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
        closers = register_message_routes(
            dapr_client=mock_client, targets=[handler], loop=loop
        )
    finally:
        loop.close()

    # Should create two subscriptions
    assert mock_client.subscribe_with_handler.call_count == 2
    assert len(closers) == 2

    # Verify both topics were registered
    topics = [
        call.kwargs["topic"]
        for call in mock_client.subscribe_with_handler.call_args_list
    ]
    assert "orders.created" in topics
    assert "orders.cancelled" in topics


def test_register_message_handlers_ignores_undecorated_methods():
    """Test that methods without @message_router are ignored."""
    mock_client = MagicMock()
    mock_client.subscribe_with_handler.return_value = MagicMock()

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
        closers = register_message_routes(
            dapr_client=mock_client, targets=[handler], loop=loop
        )
    finally:
        loop.close()

    # Should only create one subscription (for decorated method)
    assert mock_client.subscribe_with_handler.call_count == 1
    assert len(closers) == 1


def test_register_message_handlers_handles_multiple_targets():
    """Test registering multiple targets (functions and instances)."""
    mock_client = MagicMock()
    mock_client.subscribe_with_handler.return_value = MagicMock()

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
        closers = register_message_routes(
            dapr_client=mock_client,
            targets=[standalone_handler, handler_instance],
            loop=loop,
        )
    finally:
        loop.close()

    # Should create two subscriptions
    assert mock_client.subscribe_with_handler.call_count == 2
    assert len(closers) == 2


def test_register_message_handlers_returns_closers():
    """Test that closer functions are returned for each subscription."""
    mock_client = MagicMock()
    mock_client.subscribe_with_handler.return_value = MagicMock()

    @message_router(pubsub="messagepubsub", topic="orders.created")
    def handle_created(message: OrderCreated):
        pass

    @message_router(pubsub="messagepubsub", topic="orders.cancelled")
    def handle_cancelled(message: OrderCancelled):
        pass

    loop = asyncio.new_event_loop()
    try:
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
