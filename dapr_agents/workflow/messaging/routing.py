import asyncio
import inspect
import logging
import threading
import functools
from typing import Callable

from dapr.aio.clients.grpc.subscription import Subscription
from dapr.clients.grpc._response import TopicEventResponse
from dapr.clients.grpc.subscription import StreamInactiveError
from dapr.common.pubsub.subscription import StreamCancelledError, SubscriptionMessage
from dapr_agents.workflow.messaging.parser import (
    extract_cloudevent_data,
    validate_message_model,
)
from dapr_agents.workflow.messaging.utils import is_valid_routable_model
from dapr_agents.workflow.utils import get_decorated_methods

logger = logging.getLogger(__name__)


class MessageRoutingMixin:
    """
    Mixin class providing dynamic message routing capabilities for agentic services using Dapr pub/sub.

    This mixin enables:
    - Auto-registration of message handlers via the `@message_router` decorator.
    - CloudEvent-based dispatch to appropriate handlers based on `type`.
    - Topic subscription management and graceful shutdown.
    - Support for both synchronous and asynchronous handler methods.
    - Workflow-aware message handling for registered workflow entrypoints.

    Expected attributes provided by the consuming service:
    - `self._dapr_client`: A configured Dapr client instance.
    - `self.name`: The agent's name (used for default topic routing).
    - `self.message_bus_name`: Pub/Sub component name in Dapr.
    - `self.broadcast_topic_name`: Optional default topic name for broadcasts.
    - `self._topic_handlers`: Dict storing routing info by (pubsub, topic).
    - `self._subscriptions`: Dict storing unsubscribe functions for active subscriptions.
    """

    def register_message_routes(self) -> None:
        """
        Registers message handlers dynamically by subscribing once per topic.
        Incoming messages are dispatched by CloudEvent `type` to the appropriate handler.

        This function:
        - Scans all class methods for the `@message_router` decorator.
        - Extracts routing metadata and message model schemas.
        - Wraps each handler and maps it by `(pubsub_name, topic_name)` and schema name.
        - Ensures only one handler per schema per topic is allowed.
        """
        message_handlers = get_decorated_methods(self, "_is_message_handler")

        for method_name, method in message_handlers.items():
            try:
                router_data = method._message_router_data.copy()
                pubsub_name = router_data.get("pubsub") or self.message_bus_name
                is_broadcast = router_data.get("is_broadcast", False)
                topic_name = router_data.get("topic") or (
                    self.broadcast_topic_name if is_broadcast else self.name
                )
                message_schemas = router_data.get("message_schemas", [])

                if not message_schemas:
                    raise ValueError(
                        f"No message models found for handler '{method_name}'."
                    )

                wrapped_method = self._create_wrapped_method(method)
                topic_key = (pubsub_name, topic_name)

                self._topic_handlers.setdefault(topic_key, {})

                for schema in message_schemas:
                    if not is_valid_routable_model(schema):
                        raise ValueError(
                            f"Unsupported message model for handler '{method_name}': {schema}"
                        )

                    schema_name = schema.__name__
                    logger.debug(
                        f"Registering handler '{method_name}' for topic '{topic_name}' with model '{schema_name}'"
                    )

                    # Prevent multiple handlers for the same schema
                    if schema_name in self._topic_handlers[topic_key]:
                        raise ValueError(
                            f"Duplicate handler for model '{schema_name}' on topic '{topic_name}'. "
                            f"Each model can only be handled by one function per topic."
                        )

                    self._topic_handlers[topic_key][schema_name] = {
                        "schema": schema,
                        "handler": wrapped_method,
                    }

            except Exception as e:
                logger.error(
                    f"Failed to register handler '{method_name}': {e}", exc_info=True
                )

        # Subscribe once per topic
        for pubsub_name, topic_name in self._topic_handlers.keys():
            self._subscribe_with_router(pubsub_name, topic_name)

        logger.info("All message routes registered.")

    def _create_wrapped_method(self, method: Callable) -> Callable:
        """
        Wraps a message handler method to ensure it runs asynchronously,
        with special handling for workflows.
        """

        @functools.wraps(method)
        async def wrapped_method(message: dict):
            try:
                if getattr(method, "_is_workflow", False):
                    workflow_name = getattr(method, "_workflow_name", method.__name__)
                    instance_id = self.run_workflow(workflow_name, input=message)
                    asyncio.create_task(self.monitor_workflow_completion(instance_id))
                    return None

                if inspect.iscoroutinefunction(method):
                    return await method(message=message)
                else:
                    return method(message=message)

            except Exception as e:
                logger.error(
                    f"Error invoking handler '{method.__name__}': {e}", exc_info=True
                )
                return None

        return wrapped_method

    def _subscribe_with_router(self, pubsub_name: str, topic_name: str):
        subscription: Subscription = self._dapr_client.subscribe(
            pubsub_name, topic_name
        )
        loop = asyncio.get_running_loop()

        def stream_messages(sub: Subscription):
            while True:
                try:
                    for message in sub:
                        if message:
                            try:
                                future = asyncio.run_coroutine_threadsafe(
                                    self._route_message(
                                        pubsub_name, topic_name, message
                                    ),
                                    loop,
                                )
                                response = future.result()
                                sub.respond(message, response.status)
                            except Exception as e:
                                print(f"Error handling message: {e}")
                        else:
                            continue
                except (StreamInactiveError, StreamCancelledError):
                    break

        def close_subscription():
            subscription.close()

        self._subscriptions[(pubsub_name, topic_name)] = close_subscription
        threading.Thread(
            target=stream_messages, args=(subscription,), daemon=True
        ).start()

    # TODO: retry setup should be configurable
    async def _route_message(
        self, pubsub_name: str, topic_name: str, message: SubscriptionMessage
    ) -> TopicEventResponse:
        """
        Routes an incoming message to the correct handler based on CloudEvent `type`.

        Args:
            pubsub_name (str): The name of the pubsub component.
            topic_name (str): The topic from which the message was received.
            message (SubscriptionMessage): The incoming Dapr message.

        Returns:
            TopicEventResponse: The response status for the message (success, drop, retry).
        """
        try:
            handler_map = self._topic_handlers.get((pubsub_name, topic_name), {})
            if not handler_map:
                logger.warning(
                    f"No handlers for topic '{topic_name}' on pubsub '{pubsub_name}'. Dropping message."
                )
                return TopicEventResponse("drop")

            # Step 1: Extract CloudEvent metadata and data
            event_data, metadata = extract_cloudevent_data(message)
            event_type = metadata.get("type")

            route_entry = handler_map.get(event_type)
            if not route_entry:
                logger.warning(
                    f"No handler matched CloudEvent type '{event_type}' on topic '{topic_name}'"
                )
                return TopicEventResponse("drop")

            schema = route_entry["schema"]
            handler = route_entry["handler"]

            try:
                parsed_message = validate_message_model(schema, event_data)
                parsed_message["_message_metadata"] = metadata

                logger.info(
                    f"Dispatched to handler '{handler.__name__}' for event type '{event_type}'"
                )
                result = await handler(parsed_message)
                if result is not None:
                    return TopicEventResponse("success"), result

                return TopicEventResponse("success")

            except Exception as e:
                logger.warning(
                    f"Failed to validate message against schema '{schema.__name__}': {e}"
                )
                return TopicEventResponse("retry")

        except Exception as e:
            logger.error(f"Unexpected error during message routing: {e}", exc_info=True)
            return TopicEventResponse("retry")
