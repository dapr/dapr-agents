from dapr_agents.memory import MemoryBase, ConversationListMemory, ConversationVectorMemory
from dapr_agents.storage.daprstores.statestore import DaprStateStore
from dapr_agents.agent.utils.text_printer import ColorTextFormatter
from dapr_agents.messaging import DaprPubSub, parse_cloudevent
from dapr_agents.workflow.service import WorkflowAppService
from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Any, Optional, Union, Dict, Type, List
from fastapi import Depends, Request, Response, HTTPException
from dapr.clients import DaprClient
import tempfile
import json
import threading
import inspect
import logging
import asyncio
import json
import os

state_lock = threading.Lock()

logger = logging.getLogger(__name__)

class AgenticWorkflowService(WorkflowAppService, DaprPubSub):
    """
    A service class for managing agentic workflows, extending `WorkflowAppService`.
    Handles agent interactions, workflow execution, and metadata management.
    """

    name: str = Field(..., description="The name of the agentic system.")
    message_bus_name: str = Field(..., description="The name of the Dapr message bus component, defining the pub/sub base.")
    broadcast_topic_name: Optional[str] = Field("beacon_channel", description="Default topic for broadcasting messages to all agents.")
    state_store_name: str = Field(..., description="Dapr state store for agentic workflow state.")
    state_key: str = Field(default="workflow_state", description="Dapr state store key for agentic workflow state.")
    state: Optional[Union[BaseModel, dict]] = Field(default=None, description="The current state of the workflow. If not provided, it is loaded from storage if available; otherwise, it initializes an empty state.")
    state_format: Optional[Type[BaseModel]] = Field(default=None, description="The schema to enforce state structure and validation.")
    agents_registry_store_name: str = Field(..., description="Dapr state store component for centralized agent metadata.")
    agents_registry_key: str = Field(default="agents_registry", description="Dapr state store key for agentic workflow state.")
    max_iterations: int = Field(default=20, description="Maximum number of iterations for workflows. Must be greater than 0.", ge=1)
    memory: MemoryBase = Field(default_factory=ConversationListMemory, description="Handles conversation history and context storage.")
    save_state_locally: bool = Field(default=True, description="Whether to save the workflow state locally as a backup after each update. Defaults to True.")
    local_state_path: Optional[str] = Field(default=None, description="The file path where the workflow state is saved locally. If not provided, defaults to the directory where the service is instantiated.")

    # Fields Initialized during class initialization
    state_store_client: Optional[DaprStateStore] = Field(default=None, init=False, description="Dapr state store instance for accessing and managing workflow state.")
    text_formatter: Optional[ColorTextFormatter] = Field(default=None, init=False, description="Formatter for colored text output.")
    agent_metadata: Optional[Dict[str, Any]] = Field(default=None, init=False, description="Dictionary containing metadata of the agent.")

    @model_validator(mode="before")
    def set_service_name(cls, values: dict) -> dict:
        """
        Ensures that `service_name` is set based on the name of the agentic system (`name`) if it is not explicitly provided.

        Args:
            values (dict): Dictionary of field values before model validation.
        
        Returns:
            dict: The updated values with `service_name` set if it was missing.
        """
        if not values.get("service_name") and values.get("name"):
            values["service_name"] = values["name"]

        return values
    
    def model_post_init(self, __context: Any) -> None:
        """
        Configure agentic workflows, set state parameters, and initialize metadata store.
        """

        # Initialize Text Formatter
        self.text_formatter = ColorTextFormatter()

        # Initialize Workflow state store client only if a state store name is provided
        self.state_store_client = DaprStateStore(store_name=self.state_store_name, address=self.daprGrpcAddress)
        logger.info(f"State store '{self.state_store_name}' initialized.")
        
        # Load or initialize state
        self.initialize_state()

        # Initialize WorkflowAppService (parent class)
        super().model_post_init(__context)

        # Registering App routes and subscriping to topics dynamically
        self.register_message_routes()

        # Adding other API Routes
        self.app.add_api_route("/GetChatHistory", self.get_chat_history, methods=["GET"])
    
    def get_chat_history(self, task: Optional[str] = None) -> List[dict]:
        """
        Retrieves and validates the agent's chat history.

        This function fetches messages stored in the agent's memory, optionally filtering
        them based on the given task using vector similarity. The retrieved messages are
        validated using Pydantic (if applicable) and returned as a list of dictionaries.

        Args:
            task (str, optional): A specific task description to filter relevant messages 
                using vector embeddings. If not provided, retrieves the full chat history.

        Returns:
            List[dict]: A list of chat history messages, each represented as a dictionary.
                If a message is a Pydantic model, it is serialized using `model_dump()`.
        """
        if isinstance(self.memory, ConversationVectorMemory) and task:
            query_embeddings = self.memory.vector_store.embedding_function.embed(task)
            chat_history = self.memory.get_messages(query_embeddings=query_embeddings)
        else:
            chat_history = self.memory.get_messages()
        chat_history_messages = [msg.model_dump() if isinstance(msg, BaseModel) else msg for msg in chat_history]
        return chat_history_messages

    def initialize_state(self) -> None:
        """
        Initializes the workflow state by using a provided state, loading from storage, or setting an empty state.

        If the user provides a state, it is validated and used. Otherwise, the method attempts to load 
        the existing state from storage. If no stored state is found, an empty dictionary is initialized.

        Ensures `self.state` is always a valid dictionary. If a state format (`self.state_format`) 
        is provided, the structure is validated before saving.

        Raises:
            TypeError: If `self.state` is not a dictionary or a valid Pydantic model.
            RuntimeError: If state initialization or loading from storage fails.
        """
        try:
            # Load from storage if the user didn't provide a state
            if self.state is None:
                logger.info("No user-provided state. Attempting to load from storage.")
                self.state = self.load_state()

            # Ensure state is a valid dictionary or Pydantic model
            if isinstance(self.state, BaseModel):
                logger.info("User provided a state as a Pydantic model. Converting to dict.")
                self.state = self.state.model_dump()

            if not isinstance(self.state, dict):
                raise TypeError(f"Invalid state type: {type(self.state)}. Expected dict or Pydantic model.")

            logger.info(f"Workflow state initialized with {len(self.state)} key(s).")
            self.save_state()

        except Exception as e:
            logger.error(f"Failed to initialize workflow state: {e}")
            raise RuntimeError(f"Error initializing workflow state: {e}") from e
    
    def validate_state(self, state_data: dict) -> dict:
        """
        Validates the workflow state against the defined schema (`state_format`).

        If a `state_format` (Pydantic model) is provided, this method ensures that 
        the `state_data` conforms to the expected structure. If validation succeeds, 
        it returns the structured state as a dictionary.

        Args:
            state_data (dict): The raw state data to validate.

        Returns:
            dict: The validated and structured state.

        Raises:
            ValidationError: If the state data does not conform to the expected schema.
        """
        try:
            if not self.state_format:
                logger.warning("No schema (state_format) provided; returning state as-is.")
                return state_data

            logger.info("Validating workflow state against schema.")
            validated_state: BaseModel = self.state_format(**state_data)  # Validate with Pydantic
            return validated_state.model_dump()  # Convert validated model to dict

        except ValidationError as e:
            logger.error(f"State validation failed: {e}")
            raise ValidationError(f"Invalid workflow state: {e.errors()}") from e
    
    def load_state(self) -> dict:
        """
        Loads the workflow state from the Dapr state store.

        This method attempts to retrieve the stored state from the configured Dapr state store. 
        If no state exists in storage, it initializes an empty state.

        Returns:
            dict: The loaded and optionally validated state.

        Raises:
            RuntimeError: If the state store is not properly configured.
            TypeError: If the retrieved state is not a dictionary.
            ValidationError: If state schema validation fails.
        """
        try:
            if not self.state_store_client or not self.state_store_name or not self.state_key:
                logger.error("State store is not configured. Cannot load state.")
                raise RuntimeError("State store is not configured. Please provide 'state_store_name' and 'state_key'.")

            # Avoid overwriting state if self.state is already set
            if self.state:
                logger.info("Using existing in-memory state. Skipping load from storage.")
                return self.state

            has_state, state_data = self.state_store_client.try_get_state(self.state_key)

            if has_state and state_data:
                logger.info(f"Existing state found for key '{self.state_key}'. Validating it.")

                if not isinstance(state_data, dict):
                    raise TypeError(f"Invalid state type retrieved: {type(state_data)}. Expected dict.")

                return self.validate_state(state_data) if self.state_format else state_data

            logger.info(f"No existing state found for key '{self.state_key}'. Initializing empty state.")
            return {}

        except Exception as e:
            logger.error(f"Failed to load state for key '{self.state_key}': {e}")
            raise RuntimeError(f"Error loading workflow state: {e}") from e
    
    def get_local_state_file_path(self) -> str:
        """
        Returns the file path for saving the local state.
        
        If `local_state_path` is None, it defaults to the current working directory with a filename based on `state_key`.
        """
        directory = self.local_state_path or os.getcwd()
        os.makedirs(directory, exist_ok=True)  # Ensure directory exists
        return os.path.join(directory, f"{self.state_key}.json")

    def save_state_to_disk(self, state_data: str, filename: Optional[str] = None) -> None:
        """
        Safely saves the workflow state to a local JSON file using a uniquely named temp file.
        - Writes to a temp file in parallel.
        - Locks only the final atomic replacement step to avoid overwriting.
        """
        try:
            # Determine save location
            save_directory = self.local_state_path or os.getcwd()
            os.makedirs(save_directory, exist_ok=True)  # Ensure directory exists
            filename = filename or f"{self.name}_state.json"
            file_path = os.path.join(save_directory, filename)

            # Write to a uniquely named temp file
            with tempfile.NamedTemporaryFile("w", dir=save_directory, delete=False) as tmp_file:
                tmp_file.write(state_data)
                temp_path = tmp_file.name  # Save temp file path

            # Lock only for the final atomic file replacement
            with state_lock:
                # Load the existing state (merge changes)
                existing_state = {}
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as file:
                        try:
                            existing_state = json.load(file)
                        except json.JSONDecodeError:
                            logger.warning("Existing state file is corrupt or empty. Overwriting.")

                # Merge new state into existing state
                new_state = json.loads(state_data) if isinstance(state_data, str) else state_data
                merged_state = {**existing_state, **new_state}  # Merge updates

                # Write merged state back to a new temp file
                with open(temp_path, "w", encoding="utf-8") as file:
                    json.dump(merged_state, file, indent=4)

                # Atomically replace the old state file
                os.replace(temp_path, file_path)

            logger.info(f"Workflow state saved locally at '{file_path}'.")

        except Exception as e:
            logger.error(f"Failed to save workflow state to disk: {e}")
            raise RuntimeError(f"Error saving workflow state to disk: {e}")
    
    def save_state(self, state: Optional[Union[dict, BaseModel, str]] = None, force_reload: bool = False) -> None:
        """
        Saves the current workflow state to the Dapr state store and optionally as a local backup.

        This method updates the internal `self.state`, serializes it, and persists it to Dapr's state store.
        If `save_state_locally` is `True`, it calls `save_state_to_disk` to write the state to a local file.

        Args:
            state (Optional[Union[dict, BaseModel, str]], optional): 
                The new state to save. If not provided, the method saves the existing `self.state`.
            force_reload (bool, optional): 
                If `True`, reloads the state from the store after saving to ensure consistency.
                Defaults to `False`.

        Raises:
            RuntimeError: If the state store is not configured.
            TypeError: If the provided state is not a supported type (dict, BaseModel, or JSON string).
            ValueError: If the provided state is a string but not a valid JSON format.
            Exception: If any error occurs during the save operation.
        """
        try:
            if not self.state_store_client or not self.state_store_name or not self.state_key:
                logger.error("State store is not configured. Cannot save state.")
                raise RuntimeError("State store is not configured. Please provide 'state_store_name' and 'state_key'.")

            # Update self.state with the new state if provided
            self.state = state or self.state
            if not self.state:
                logger.warning("Skipping state save: Empty state.")
                return

            # Convert state to a JSON-compatible format
            if isinstance(self.state, BaseModel):
                state_to_save = self.state.model_dump_json()  
            elif isinstance(self.state, dict):
                state_to_save = json.dumps(self.state)  
            elif isinstance(self.state, str):
                try:
                    json.loads(self.state)  # Ensure the string is valid JSON
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON string provided as state: {e}")
                state_to_save = self.state
            else:
                raise TypeError(f"Invalid state type: {type(self.state)}. Expected dict, BaseModel, or JSON string.")

            # Save state in Dapr
            self.state_store_client.save_state(self.state_key, state_to_save)
            logger.info(f"Successfully saved state for key '{self.state_key}'.")

            # Save state locally if enabled
            if self.save_state_locally:
                self.save_state_to_disk(state_data=state_to_save)

            # Reload state after saving if requested
            if force_reload:
                self.state = self.load_state()
                logger.info(f"State reloaded after saving for key '{self.state_key}'.")

        except Exception as e:
            logger.error(f"Failed to save state for key '{self.state_key}': {e}")
            raise
    
    def get_agents_metadata(self, exclude_self: bool = True, exclude_orchestrator: bool = False) -> dict:
        """
        Retrieves metadata for all registered agents while ensuring orchestrators do not interact with other orchestrators.

        Args:
            exclude_self (bool, optional): If True, excludes the current agent (`self.name`). Defaults to True.
            exclude_orchestrator (bool, optional): If True, excludes all orchestrators from the results. Defaults to False.

        Returns:
            dict: A mapping of agent names to their metadata. Returns an empty dict if no agents are found.

        Raises:
            RuntimeError: If the state store is not properly configured or retrieval fails.
        """
        try:
            # Fetch agent metadata
            agents_metadata = self.get_data_from_store(self.agents_registry_store_name, self.agents_registry_key) or {}

            if agents_metadata:
                logger.info(f"Agents found in '{self.agents_registry_store_name}' for key '{self.agents_registry_key}'.")

                # Filter based on self-exclusion and orchestrator exclusion
                filtered_metadata = {
                    name: metadata
                    for name, metadata in agents_metadata.items()
                    if not (exclude_self and name == self.name)  # Exclude self if requested
                    and not (exclude_orchestrator and metadata.get("orchestrator", False))  # Exclude orchestrators only if exclude_orchestrator=True
                }

                if not filtered_metadata:
                    logger.info("No other agents found after filtering.")

                return filtered_metadata

            logger.info(f"No agents found in '{self.agents_registry_store_name}' for key '{self.agents_registry_key}'.")
            return {}
        except Exception as e:
            logger.error(f"Failed to retrieve agents metadata: {e}", exc_info=True)
            raise RuntimeError(f"Error retrieving agents metadata: {str(e)}") from e
    
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

            logger.info(f"{self.name} broadcasting message to selected agents.")

            await self.publish_event_message(
                topic_name=self.broadcast_topic_name,
                pubsub_name=self.message_bus_name,
                source=self.name,
                message=message,
                **kwargs,
            )

            logger.debug(f"{self.name} broadcasted message.")
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
            agents_metadata = self.get_agents_metadata()
            if name not in agents_metadata:
                logger.error(f"Agent {name} not found.")
                raise RuntimeError(f"Agent {name} not found.")

            agent_metadata = agents_metadata[name]
            logger.info(f"{self.name} sending message to agent '{name}'.")

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
            raise HTTPException(status_code=500, detail=f"Error sending message to agent '{name}': {str(e)}")
    
    def print_interaction(self, sender_agent_name: str, recipient_agent_name: str, message: str) -> None:
        """
        Prints the interaction between two agents in a formatted and colored text.

        Args:
            sender_agent_name (str): The name of the agent sending the message.
            recipient_agent_name (str): The name of the agent receiving the message.
            message (str): The message content to display.
        """
        separator = "-" * 80
        
        # Print sender -> recipient and the message
        interaction_text = [
            (sender_agent_name, "dapr_agents_mustard"),
            (" -> ", "dapr_agents_teal"),
            (f"{recipient_agent_name}\n\n", "dapr_agents_mustard"),
            (message + "\n\n", None),
            (separator + "\n", "dapr_agents_teal"),
        ]

        # Print the formatted text
        self.text_formatter.print_colored_text(interaction_text)
    
    def register_agentic_system(self) -> None:
        """
        Registers the agent's metadata in the Dapr state store under 'agents_metadata'.
        """
        try:
            # Retrieve existing metadata (always returns a dict)
            agents_metadata = self.get_agents_metadata()
            agents_metadata[self.name] = self.agent_metadata

            # Save the updated metadata back to Dapr store
            with DaprClient(address=self.daprGrpcAddress) as client:
                client.save_state(
                    store_name=self.agents_registry_store_name,
                    key=self.agents_registry_key,
                    value=json.dumps(agents_metadata),
                    state_metadata={"contentType": "application/json"}
                )

            logger.info(f"{self.name} registered its metadata under key '{self.agents_registry_key}'")

        except Exception as e:
            logger.error(f"Failed to register metadata for agent {self.name}: {e}")
    
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
                
                # Extract message and metadata from the request
                message, metadata = await parse_cloudevent(request, model)

                # Return the message along with metadata in a dictionary
                return {"message": message, "metadata": metadata}
            return dependency_injector

        def create_wrapped_method(method):
            """Wraps method to handle workflows and regular handlers differently."""
            async def wrapped_method(dependencies: dict = Depends(dependency_injector)):
                try:
                    # Validate expected parameters
                    handler_signature = inspect.signature(method)
                    expected_params = {
                        key: value for key, value in dependencies.items()
                        if key in handler_signature.parameters
                    }

                    if hasattr(method, "_is_workflow"):
                        workflow_name = getattr(method, "_workflow_name")

                        # Identify method parameters, skipping 'self' and 'cls'
                        workflow_params = list(handler_signature.parameters.keys())
                        filtered_params = [p for p in workflow_params if p not in {"self", "cls"}]

                        # Find the input parameter after ctx
                        input_param_name = filtered_params[1] if len(filtered_params) > 1 else None

                        # Ensure input is serializable
                        input_data = expected_params.get(input_param_name)  # Retrieve input dynamically
                        if isinstance(input_data, BaseModel):
                            input_data = input_data.model_dump()  # Convert Pydantic model to dict
                        
                        # Ensure metadata is serializable
                        metadata = dependencies.get("metadata")  # Retrieve metadata from dependencies directly
                        if isinstance(metadata, BaseModel):
                            metadata = metadata.model_dump() # Convert Pydantic model to dict

                        # Explicitly attach metadata to input before sending to workflow
                        if isinstance(input_data, dict):
                            input_data["_message_metadata"] = metadata  # Ensure metadata is passed along

                        logger.info(f"Starting workflow '{workflow_name}'")
                        logger.debug(f"Workflow Input: {input_data}")

                        # Run workflow and monitor completion
                        instance_id = self.run_workflow(workflow_name, input=input_data)
                        asyncio.create_task(self.monitor_workflow_completion(instance_id))

                        return Response(content=f"Workflow '{workflow_name}' started. Instance ID: {instance_id}", status_code=202)

                    # Handle regular message handlers (non-workflow functions)
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
                    topic_name = router_data.get("topic") or (self.name if not is_broadcast else self.broadcast_topic_name)
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