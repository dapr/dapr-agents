import logging
import json
import asyncio
import inspect
import threading
import functools
from dataclasses import is_dataclass, asdict
from typing import Optional, Any, Dict, Union, Callable

from pydantic import BaseModel

from dapr.aio.clients import DaprClient
from dapr.aio.clients.grpc.subscription import Subscription
from dapr.clients.grpc._response import TopicEventResponse
from dapr.clients.grpc.subscription import StreamInactiveError
from dapr.common.pubsub.subscription import StreamCancelledError, SubscriptionMessage
from dapr_agents.workflow.utils.messaging import (
    extract_cloudevent_data,
    validate_message_model,
)
from dapr_agents.workflow.utils.core import (
    get_decorated_methods,
    is_pydantic_model,
    is_valid_routable_model,
)

logger = logging.getLogger(__name__)


class PubSubMixin:
    """
    Mixin providing Dapr-based pub/sub messaging, event publishing, and dynamic message routing.

    Features:
        - Publishes messages and events to Dapr topics with optional CloudEvent metadata.
        - Registers message handlers dynamically using decorated methods.
        - Routes incoming messages to handlers based on CloudEvent `type` and message schema.
        - Supports Pydantic models, dataclasses, and dictionaries as message payloads.
        - Handles asynchronous message processing and workflow invocation.
        - Manages topic subscriptions and message dispatch via Dapr client.
    """

    async def serialize_message(self, message: Any) -> str:
        """
        Serializes a message to JSON format.

        Args:
            message (Any): The message content to serialize.

        Returns:
            str: JSON string of the message.

        Raises:
            ValueError: If the message is not serializable.
        """
        try:
            return json.dumps(message if message is not None else {})
        except TypeError as te:
            logger.error(f"Failed to serialize message: {message}. Error: {te}")
            raise ValueError(f"Message contains non-serializable data: {te}")

    async def publish_message(
        self,
        pubsub_name: str,
        topic_name: str,
        message: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Publishes a message to a specific topic with optional metadata.

        Args:
            pubsub_name (str): The pub/sub component to use.
            topic_name (str): The topic to publish the message to.
            message (Any): The message content, can be None or any JSON-serializable type.
            metadata (Optional[Dict[str, Any]]): Additional metadata to include in the publish event.

        Raises:
            ValueError: If the message contains non-serializable data.
            Exception: If publishing the message fails.
        """
        try:
            json_message = await self.serialize_message(message)

            # TODO: retry publish should be configurable
            async with DaprClient() as client:
                await client.publish_event(
                    pubsub_name=pubsub_name or self.message_bus_name,
                    topic_name=topic_name,
                    data=json_message,
                    data_content_type="application/json",
                    publish_metadata=metadata or {},
                )

            logger.debug(
                f"Message successfully published to topic '{topic_name}' on pub/sub '{pubsub_name}'."
            )
            logger.debug(f"Serialized Message: {json_message}, Metadata: {metadata}")
        except Exception as e:
            logger.error(
                f"Error publishing message to topic '{topic_name}' on pub/sub '{pubsub_name}'. "
                f"Message: {message}, Metadata: {metadata}, Error: {e}"
            )
            raise Exception(
                f"Failed to publish message to topic '{topic_name}' on pub/sub '{pubsub_name}': {str(e)}"
            )

    async def publish_event_message(
        self,
        topic_name: str,
        pubsub_name: str,
        source: str,
        message: Union[BaseModel, dict, Any],
        message_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Publishes an event message to a specified topic with dynamic metadata.

        Args:
            topic_name (str): The topic to publish the message to.
            pubsub_name (str): The pub/sub component to use.
            source (str): The source of the message (e.g., service or agent name).
            message (Union[BaseModel, dict, dataclass, Any]): The message content, as a Pydantic model, dictionary, or dataclass instance.
            message_type (Optional[str]): The type of the message. Required if `message` is a dictionary.
            **kwargs: Additional metadata fields to include in the message.
        """
        if isinstance(message, BaseModel):
            message_type = message_type or message.__class__.__name__
            message_dict = message.model_dump()

        elif isinstance(message, dict):
            if not message_type:
                raise ValueError(
                    "message_type must be provided when message is a dictionary."
                )
            message_dict = message

        elif is_dataclass(message):
            message_type = message_type or message.__class__.__name__
            message_dict = asdict(message)

        else:
            raise ValueError(
                "Message must be a Pydantic BaseModel, a dictionary, or a dataclass instance."
            )

        metadata = {
            "cloudevent.type": message_type,
            "cloudevent.source": source,
        }
        metadata.update(kwargs)

        logger.debug(
            f"{source} preparing to publish '{message_type}' to topic '{topic_name}'."
        )
        logger.debug(f"Message: {message_dict}, Metadata: {metadata}")

        await self.publish_message(
            topic_name=topic_name,
            pubsub_name=pubsub_name or self.message_bus_name,
            message=message_dict,
            metadata=metadata,
        )

        logger.info(f"{source} published '{message_type}' to topic '{topic_name}'.")

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
            if topic_name:
                # Prevent subscribing to empty or None topics
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
                is_workflow = getattr(method, "_is_workflow", False)
                message_type = (
                    type(message).__name__
                    if hasattr(message, "__class__")
                    else str(type(message))
                )
                logger.debug(
                    f"PubSub routing for {method.__name__}: _is_workflow={is_workflow}, message_type={message_type}"
                )
                if is_workflow:
                    workflow_name = getattr(method, "_workflow_name", method.__name__)
                    # If the message is a Pydantic model, extract metadata and convert to dict
                    if is_pydantic_model(type(message)):
                        # Extract metadata if available
                        metadata = getattr(message, "_message_metadata", None)
                        # Convert to dict for workflow input
                        message_dict = message.model_dump()
                        if metadata is not None:
                            # Include metadata in the message dict
                            message_dict["_message_metadata"] = metadata
                        message = message_dict

                    # Prevent triggering multiple orchestrator workflows if one is already running
                    if (
                        workflow_name == "OrchestratorWorkflow"
                        or workflow_name == "main_workflow"
                    ):
                        triggering_workflow_id = message.get("workflow_instance_id")
                        if triggering_workflow_id:
                            if hasattr(
                                self, "_does_workflow_exist"
                            ) and self._does_workflow_exist(triggering_workflow_id):
                                logger.info(
                                    f"Triggering workflow {triggering_workflow_id} is still running. Skipping new orchestrator instance."
                                )
                                return None

                    # Invoke the workflow
                    await self.run_and_monitor_workflow_async(
                        workflow_name, input=message
                    )
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

            # Step 2: Find the handler for the event type
            route_entry = handler_map.get(event_type)
            if not route_entry:
                # If no handler matches the event type, log and drop the message
                logger.warning(
                    f"No handler matched CloudEvent type '{event_type}' on topic '{topic_name}'"
                )
                return TopicEventResponse("drop")

            schema = route_entry["schema"]
            handler = route_entry["handler"]

            try:
                # Step 3: Validate the message against the schema
                parsed_message = validate_message_model(schema, event_data)
                # Step 4: Attach metadata to the parsed message
                if isinstance(parsed_message, dict):
                    parsed_message["_message_metadata"] = metadata
                else:
                    setattr(parsed_message, "_message_metadata", metadata)

                logger.info(
                    f"Dispatched to handler '{handler.__name__}' for event type '{event_type}'"
                )
                # Step 5: Call the handler with the parsed message
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
