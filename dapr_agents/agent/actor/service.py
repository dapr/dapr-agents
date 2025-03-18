from dapr.actor.runtime.config import ActorRuntimeConfig, ActorTypeConfig, ActorReentrancyConfig
from dapr.clients.grpc._response import StateResponse
from dapr.actor.runtime.runtime import ActorRuntime
from dapr.actor import ActorProxy, ActorId
from dapr.aio.clients import DaprClient
from dapr.ext.fastapi import DaprActor
from fastapi import FastAPI, Depends, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from dapr_agents.messaging import parse_cloudevent
from dapr_agents.agent.actor import AgentActorBase, AgentActorInterface
from dapr_agents.service.fastapi import DaprFastAPIServer
from dapr_agents.messaging import DaprPubSub
from dapr_agents.types.agent import AgentActorMessage
from dapr_agents.agent import AgentBase
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional, Any, Union
from contextlib import asynccontextmanager
from datetime import timedelta
from inspect import signature
import json
import logging

logger = logging.getLogger(__name__)

class AgentActorServiceBase(DaprFastAPIServer, DaprPubSub):
    """
    A Pydantic-based class for managing services and exposing FastAPI routes with Dapr pub/sub and actor support.
    """

    agent: AgentBase
    agent_topic_name: Optional[str] = Field(None, description="The topic name dedicated to this specific agent, derived from the agent's name if not provided.")
    broadcast_topic_name: str = Field("beacon_channel", description="The default topic used for broadcasting messages to all agents.")
    agents_registry_store_name: str = Field(..., description="The name of the Dapr state store component used to store and share agent metadata centrally.")
    agents_registry_key: str = Field(default="agents_registry", description="Dapr state store key for agentic workflow state.")

    # Fields initialized in model_post_init
    actor: Optional[DaprActor] = Field(default=None, init=False, description="DaprActor for actor lifecycle support.")
    actor_name: Optional[str] = Field(default=None, init=False, description="Actor name")
    actor_proxy: Optional[ActorProxy] = Field(default=None, init=False, description="Proxy for invoking methods on the agent's actor.")
    actor_class: Optional[type] = Field(default=None, init=False, description="Dynamically created actor class for the agent")
    agent_metadata: Optional[dict] = Field(default=None, init=False, description="Agent's metadata")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dapr_client: Optional[DaprClient] = Field(default=None, init=False, description="Dapr client for state and messaging.")

    @model_validator(mode="before")
    def set_service_name_and_topic(cls, values: dict):
        # Derive the service name from the agent's name or role
        if not values.get("service_name") and "agent" in values:
            values["service_name"] = values["agent"].name or values["agent"].role
        # Derive agent_topic_name from service name if not provided
        if not values.get("agent_topic_name") and values.get("service_name"):
            values["agent_topic_name"] = values["service_name"]
        return values

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to configure the Dapr settings, FastAPI app, and other components.
        """

        # Proceed with base model setup
        super().model_post_init(__context)

        # Dynamically create the actor class based on the agent's name
        actor_class_name = f"{self.agent.name}Actor"

        # Create the actor class dynamically using the 'type' function
        self.actor_class = type(actor_class_name, (AgentActorBase,), {
            '__init__': lambda self, ctx, actor_id: AgentActorBase.__init__(self, ctx, actor_id),
            'agent': self.agent
        })

        # Prepare agent metadata
        self.agent_metadata = {
            "name": self.agent.name,
            "role": self.agent.role,
            "goal": self.agent.goal,
            "topic_name": self.agent_topic_name,
            "pubsub_name": self.message_bus_name,
            "orchestrator": False
        }

        # Proxy for actor methods
        self.actor_name = self.actor_class.__name__
        self.actor_proxy = ActorProxy.create(self.actor_name, ActorId(self.agent.name), AgentActorInterface)
        
        # DaprActor for actor support
        self.actor = DaprActor(self.app)

        self.dapr_client = DaprClient()

        # Registering App routes and subscriping to topics dynamically
        self.register_message_routes()

        # Adding other API Routes
        self.app.add_api_route("/GetMessages", self.get_messages, methods=["GET"]) # Get messages from conversation history state

        logger.info(f"Dapr Actor class {self.actor_class.__name__} initialized.")

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """
        Extended lifespan to manage actor registration and metadata setup at startup
        and cleanup on shutdown.
        """
        # Actor Runtime Configuration (e.g., reentrancy)
        actor_runtime_config = ActorRuntimeConfig()
        actor_runtime_config.update_actor_type_configs([
            ActorTypeConfig(
                actor_type=self.actor_class.__name__,
                actor_idle_timeout=timedelta(hours=1),
                actor_scan_interval=timedelta(seconds=30),
                drain_ongoing_call_timeout=timedelta(minutes=1),
                drain_rebalanced_actors=True,
                reentrancy=ActorReentrancyConfig(enabled=True))
        ])
        ActorRuntime.set_actor_config(actor_runtime_config)
        
        # Register actor class during startup            
        await self.actor.register_actor(self.actor_class)
        logger.info(f"{self.actor_name} Dapr actor registered.")
        
        # Register agent metadata
        await self.register_agent_metadata()

        try:
            yield  # Continue with FastAPI's main lifespan context
        finally:

            # Perform any required cleanup, such as metadata removal
            await self.stop()
    
    async def get_data_from_store(self, store_name: str, key: str) -> Optional[dict]:
        """
        Retrieve data from a specified Dapr state store using a provided key.

        Args:
            store_name (str): The name of the Dapr state store component.
            key (str): The key under which the data is stored.

        Returns:
            Optional[dict]: The data stored under the specified key if found; otherwise, None.
        """
        try:
            response: StateResponse = await self.dapr_client.get_state(store_name=store_name, key=key)
            data = response.data

            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Error retrieving data for key '{key}' from store '{store_name}'")
            return None
    
    async def get_agents_metadata(self, exclude_self: bool = True, exclude_orchestrator: bool = False) -> dict:
        """
        Retrieves metadata for all registered agents while ensuring orchestrators do not interact with other orchestrators.

        Args:
            exclude_self (bool, optional): If True, excludes the current agent (`self.agent.name`). Defaults to True.
            exclude_orchestrator (bool, optional): If True, excludes all orchestrators from the results. Defaults to False.

        Returns:
            dict: A mapping of agent names to their metadata. Returns an empty dict if no agents are found.

        Raises:
            RuntimeError: If the state store is not properly configured or retrieval fails.
        """
        try:
            # Fetch agent metadata from the registry
            agents_metadata = await self.get_data_from_store(self.agents_registry_store_name, self.agents_registry_key) or {}

            if agents_metadata:
                logger.info(f"Agents found in '{self.agents_registry_store_name}' for key '{self.agents_registry_key}'.")

                # Filter based on exclusion rules
                filtered_metadata = {
                    name: metadata
                    for name, metadata in agents_metadata.items()
                    if not (exclude_self and name == self.agent.name)  # Exclude self if requested
                    and not (exclude_orchestrator and metadata.get("orchestrator", False))  # Exclude all orchestrators if exclude_orchestrator=True
                }

                if not filtered_metadata:
                    logger.info("No other agents found after filtering.")

                return filtered_metadata

            logger.info(f"No agents found in '{self.agents_registry_store_name}' for key '{self.agents_registry_key}'.")
            return {}
        except Exception as e:
            logger.error(f"Failed to retrieve agents metadata: {e}", exc_info=True)
            return {}
    
    async def register_agent_metadata(self) -> None:
        """
        Registers the agent's metadata in the Dapr state store under 'agents_metadata'.
        """
        try:
            # Retrieve existing metadata or initialize as an empty dictionary
            agents_metadata = await self.get_agents_metadata()
            agents_metadata[self.agent.name] = self.agent_metadata

            # Save the updated metadata back to Dapr store
            await self.dapr_client.save_state(
                store_name=self.agents_registry_store_name,
                key=self.agents_registry_key,
                value=json.dumps(agents_metadata),
                state_metadata={"contentType": "application/json"}
            )
            
            logger.info(f"{self.agent.name} registered its metadata under key '{self.agents_registry_key}'")

        except Exception as e:
            logger.error(f"Failed to register metadata for agent {self.agent.name}: {e}")
    
    async def invoke_task(self, task: Optional[str]) -> Response:
        """
        Use the actor to invoke a task by running the InvokeTask method through ActorProxy.

        Args:
            task (Optional[str]): The task string to invoke on the actor.

        Returns:
            Response: A FastAPI Response containing the result or an error message.
        """
        try:
            response = await self.actor_proxy.InvokeTask(task)
            return Response(content=response, status_code=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Failed to run task for {self.actor_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Error invoking task: {str(e)}")
    
    async def add_message(self, message: AgentActorMessage) -> None:
        """
        Adds a message to the conversation history in the actor's state.
        """
        try:
            await self.actor_proxy.AddMessage(message.model_dump())
        except Exception as e:
            logger.error(f"Failed to add message to {self.actor_name}: {e}")
    
    async def get_messages(self) -> Response:
        """
        Retrieve the conversation history from the actor.
        """
        try:
            messages = await self.actor_proxy.GetMessages()
            return JSONResponse(content=jsonable_encoder(messages), status_code=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Failed to retrieve messages for {self.actor_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving messages: {str(e)}")
    
    async def broadcast_message(self, message: Union[BaseModel, dict], exclude_orchestrator: bool = False, **kwargs) -> None:
        """
        Sends a message to all agents (or only to non-orchestrator agents if exclude_orchestrator=True).

        Args:
            message (Union[BaseModel, dict]): The message content as a Pydantic model or dictionary.
            exclude_orchestrator (bool, optional): If True, excludes orchestrators from receiving the message. Defaults to False.
            **kwargs: Additional metadata fields to include in the message.
        """
        try:
            # Retrieve agents metadata while respecting the exclude_orchestrator flag
            agents_metadata = await self.get_agents_metadata(exclude_orchestrator=exclude_orchestrator)

            if not agents_metadata:
                logger.warning("No agents available for broadcast.")
                return

            logger.info(f"{self.agent.name} broadcasting message to selected agents.")

            await self.publish_event_message(
                topic_name=self.broadcast_topic_name,
                pubsub_name=self.message_bus_name,
                source=self.agent.name,
                message=message,
                **kwargs,
            )

            logger.debug(f"{self.agent.name} broadcasted message.")
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error broadcasting message: {str(e)}")

    async def send_message_to_agent(self, name: str, message: Union[BaseModel, dict], **kwargs) -> None:
        """
        Sends a message to a specific agent.

        Args:
            name (str): The name of the target agent.
            message (Union[BaseModel, dict]): The message content as a Pydantic model or dictionary.
            **kwargs: Additional metadata fields to include in the message.
        """
        try:
            agents_metadata = await self.get_agents_metadata()
            if name not in agents_metadata:
                raise HTTPException(status_code=404, detail=f"Agent {name} not found.")

            agent_metadata = agents_metadata[name]
            logger.info(f"{self.agent.name} sending message to agent '{name}'.")

            await self.publish_event_message(
                topic_name=agent_metadata["topic_name"],
                pubsub_name=agent_metadata["pubsub_name"],
                source=self.agent.name,
                message=message,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Failed to send message to agent '{name}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error sending message to agent '{name}': {str(e)}")
    
    def register_message_routes(self) -> None:
        """
        Dynamically register message handlers and the Dapr /subscribe endpoint.
        """
        def register_subscription(pubsub_name: str, topic_name: str, dead_letter_topic: Optional[str] = None) -> dict:
            """Ensure a subscription exists or create it if it doesn't."""
            subscription = next(
                (
                    sub
                    for sub in self.dapr_app._subscriptions
                    if sub["pubsubname"] == pubsub_name and sub["topic"] == topic_name
                ),
                None,
            )
            if subscription is None:
                subscription = {
                    "pubsubname": pubsub_name,
                    "topic": topic_name,
                    "routes": {"rules": []},
                    **({"deadLetterTopic": dead_letter_topic} if dead_letter_topic else {}),
                }
                self.dapr_app._subscriptions.append(subscription)
                logger.info(f"Created new subscription for pubsub='{pubsub_name}', topic='{topic_name}'")
            return subscription

        def add_subscription_rule(subscription: dict, match_condition: str, route: str) -> None:
            """Add a routing rule to an existing subscription."""
            rule = {"match": match_condition, "path": route}
            if rule not in subscription["routes"]["rules"]:
                subscription["routes"]["rules"].append(rule)
                logger.info(f"Added match condition: {match_condition}")
                logger.debug(f"Rule: {rule}")
        
        def create_dependency_injector(model):
            """Factory to create a dependency injector for a specific message model."""
            async def dependency_injector(request: Request):
                if not model:
                    raise ValueError("No message model provided for dependency injection.")
                logger.debug(f"Using model '{model.__name__}' for this request.")
                message, metadata = await parse_cloudevent(request, model)
                return {"message": message, "metadata": metadata}
            return dependency_injector
        
        # Define the handler wrapper within a factory function to capture the method correctly
        def create_wrapped_method(method):
            async def wrapped_method(dependencies: dict = Depends(dependency_injector)):
                try:
                    # Validate expected parameters
                    handler_signature = signature(method)
                    expected_params = {
                        key: value for key, value in dependencies.items()
                        if key in handler_signature.parameters
                    }

                    # Call the method directly
                    result = await method(**expected_params)

                    # Wrap non-Response objects
                    if not isinstance(result, Response):
                        logger.warning("Handler returned non-Response object; wrapping it in a Response.")
                        result = Response(content=str(result), status_code=200)

                    return result
                except Exception as e:
                    logger.error(f"Error invoking handler: {e}", exc_info=True)
                    return Response(content=f"Internal Server Error: {str(e)}", status_code=500)
            return wrapped_method

        for method_name in dir(self):
            method = getattr(self, method_name, None)
            if callable(method) and hasattr(method, "_is_message_handler"):
                try:
                    # Retrieve metadata from the decorator
                    router_data = method._message_router_data.copy()
                    pubsub_name = router_data.get("pubsub") or self.message_bus_name
                    is_broadcast = router_data.get("is_broadcast", False)
                    topic_name = router_data.get("topic") or (self.agent.name if not is_broadcast else self.broadcast_topic_name)
                    route = router_data.get("route") or f"/events/{pubsub_name}/{topic_name}/{method_name}"
                    dead_letter_topic = router_data.get("dead_letter_topic")
                    message_model = router_data.get("message_model")

                    # Validate message model presence
                    if not message_model:
                        raise ValueError(f"Message model is missing for handler '{method_name}'.")

                    logger.debug(f"Registering route '{route}' for method '{method_name}' with parameters: {list(router_data.keys())}")

                    # Ensure the subscription exists and add the rule
                    subscription = register_subscription(pubsub_name, topic_name, dead_letter_topic)
                    add_subscription_rule(subscription, f"event.type == '{message_model.__name__}'", route)

                    # Create the dependency injector
                    dependency_injector = create_dependency_injector(message_model)

                    # Define the handler that wraps the original method
                    wrapped_method = create_wrapped_method(method)

                    # Register the route with the handler
                    self.app.add_api_route(route, wrapped_method, methods=["POST"], tags=["PubSub"])

                except Exception as e:
                    logger.error(f"Failed to register message route: {e}", exc_info=True)
                    raise

        logger.debug(f"Final Subscription Routes: {json.dumps(self.dapr_app._get_subscriptions(), indent=2)}")