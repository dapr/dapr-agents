import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from dapr.actor import ActorId, ActorProxy
from dapr.actor.runtime.config import (
    ActorReentrancyConfig,
    ActorRuntimeConfig,
    ActorTypeConfig,
)
from dapr.actor.runtime.runtime import ActorRuntime
from dapr.clients import DaprClient
from dapr.clients.grpc._response import StateResponse
from dapr.ext.fastapi import DaprActor

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dapr_agents.agent import AgentBase
from dapr_agents.agent.actor import AgentActorBase, AgentActorInterface
from dapr_agents.service.fastapi import FastAPIServerBase
from dapr_agents.types.agent import AgentActorMessage
from dapr_agents.workflow.messaging import DaprPubSub
from dapr_agents.workflow.messaging.routing import MessageRoutingMixin

logger = logging.getLogger(__name__)

class AgentActorService(DaprPubSub, MessageRoutingMixin):
    agent: AgentBase
    name: Optional[str] = Field(default=None, description="Name of the agent actor, derived from the agent if not provided.")
    agent_topic_name: Optional[str] = Field(None, description="The topic name dedicated to this specific agent, derived from the agent's name if not provided.")
    broadcast_topic_name: str = Field("beacon_channel", description="The default topic used for broadcasting messages to all agents.")
    agents_registry_store_name: str = Field(..., description="The name of the Dapr state store component used to store and share agent metadata centrally.")
    agents_registry_key: str = Field(default="agents_registry", description="Dapr state store key for agentic workflow state.")
    service_port: Optional[int] = Field(default=None, description="The port number to run the API server on.")
    service_host: Optional[str] = Field(default="0.0.0.0", description="Host address for the API server.")

    # Fields initialized in model_post_init
    actor: Optional[DaprActor] = Field(default=None, init=False, description="DaprActor for actor lifecycle support.")
    actor_name: Optional[str] = Field(default=None, init=False, description="Actor name")
    actor_proxy: Optional[ActorProxy] = Field(default=None, init=False, description="Proxy for invoking methods on the agent's actor.")
    actor_class: Optional[type] = Field(default=None, init=False, description="Dynamically created actor class for the agent")
    agent_metadata: Optional[dict] = Field(default=None, init=False, description="Agent's metadata")

    # Private internal attributes (not schema/validated)
    _http_server: Optional[Any] = PrivateAttr(default=None)
    _shutdown_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _dapr_client: Optional[DaprClient] = PrivateAttr(default=None)
    _is_running: bool = PrivateAttr(default=False)
    _subscriptions: Dict[str, Callable] = PrivateAttr(default_factory=dict)
    _topic_handlers: Dict[Tuple[str, str], Dict[Type[BaseModel], Callable]] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    def set_derived_fields(cls, values: dict):
        agent: AgentBase = values.get("agent")
        # Derive agent_topic_name if missing
        if not values.get("agent_topic_name") and agent:
            values["agent_topic_name"] = agent.name or agent.role
        # Derive name from agent if missing
        if not values.get("name") and agent:
            values["name"] = agent.name or agent.role
        return values
    
    def model_post_init(self, __context: Any) -> None:
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

        # Initialize Sync Dapr Client
        self._dapr_client = DaprClient()

        # FastAPI Server
        self._http_server: FastAPIServerBase = FastAPIServerBase(
            service_name=self.agent.name,
            service_port=self.service_port,
            service_host=self.service_host
        )
        self._http_server.app.router.lifespan_context = self.lifespan

        # Create DaprActor using FastAPI app
        self.actor = DaprActor(self.app)
        
        self.app.add_api_route("/GetMessages", self.get_messages, methods=["GET"])

        logger.info(f"Dapr Actor class {self.actor_class.__name__} initialized.")

    @property
    def app(self) -> "FastAPI":
        """
        Returns the FastAPI application instance if the workflow was initialized as a service.

        Raises:
            RuntimeError: If the FastAPI server has not been initialized via `.as_service()` first.
        """
        if self._http_server:
            return self._http_server.app
        raise RuntimeError("FastAPI server not initialized.")
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Register actor
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

        await self.actor.register_actor(self.actor_class)
        logger.info(f"{self.actor_name} Dapr actor registered.")

        # Register agent metadata and pubsub routes
        self.register_agent_metadata()
        self.register_message_routes()

        try:
            yield
        finally:
            await self.stop()

    async def start(self):
        if self._is_running:
            logger.warning("Service is already running. Ignoring duplicate start request.")
            return

        logger.info("Starting Agent Actor Service...")
        self._shutdown_event.clear()

        await self._http_server.start()

        self._is_running = True

    async def stop(self):
        if not self._is_running:
            return

        await self._http_server.stop()

        for (pubsub_name, topic_name), close_fn in self._subscriptions.items():
            try:
                logger.info(f"Unsubscribing from pubsub '{pubsub_name}' topic '{topic_name}'")
                close_fn()
            except Exception as e:
                logger.error(f"Failed to unsubscribe from topic '{topic_name}': {e}")

        self._subscriptions.clear()
        self._is_running = False
        logger.info("Agent Actor Service stopped.")
    
    def get_data_from_store(self, store_name: str, key: str) -> Optional[dict]:
        """
        Retrieve data from a specified Dapr state store using a provided key.

        Args:
            store_name (str): The name of the Dapr state store component.
            key (str): The key under which the data is stored.

        Returns:
            Optional[dict]: The data stored under the specified key if found; otherwise, None.
        """
        try:
            response: StateResponse = self._dapr_client.get_state(store_name=store_name, key=key)
            data = response.data

            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Error retrieving data for key '{key}' from store '{store_name}'")
            return None
    
    def get_agents_metadata(self, exclude_self: bool = True, exclude_orchestrator: bool = False) -> dict:
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
            agents_metadata = self.get_data_from_store(self.agents_registry_store_name, self.agents_registry_key) or {}

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
    
    def register_agent_metadata(self) -> None:
        """
        Registers the agent's metadata in the Dapr state store under 'agents_metadata'.
        """
        try:
            # Retrieve existing metadata or initialize as an empty dictionary
            agents_metadata = self.get_agents_metadata()
            agents_metadata[self.agent.name] = self.agent_metadata

            # Save the updated metadata back to Dapr store
            self._dapr_client.save_state(
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
            agents_metadata = self.get_agents_metadata(exclude_orchestrator=exclude_orchestrator)

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

    async def send_message_to_agent(self, name: str, message: Union[BaseModel, dict], **kwargs) -> None:
        """
        Sends a message to a specific agent.

        Args:
            name (str): The name of the target agent.
            message (Union[BaseModel, dict]): The message content as a Pydantic model or dictionary.
            **kwargs: Additional metadata fields to include in the message.
        """
        try:
            agents_metadata = self.get_agents_metadata()
            
            if name not in agents_metadata:
                logger.warning(f"Target '{name}' is not registered as an agent. Skipping message send.")
                return  # Do not raise an error—just warn and move on.

            agent_metadata = agents_metadata[name]
            logger.info(f"{self.agent.name} sending message to agent '{name}'.")

            await self.publish_event_message(
                topic_name=agent_metadata["topic_name"],
                pubsub_name=agent_metadata["pubsub_name"],
                source=self.name,
                message=message,
                **kwargs,
            )

            logger.debug(f"{self.name} sent message to agent '{name}'.")
        except Exception as e:
            logger.error(f"Failed to send message to agent '{name}': {e}", exc_info=True)