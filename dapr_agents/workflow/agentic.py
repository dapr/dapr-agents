import asyncio
import json
import logging
import os
import tempfile
import threading
import inspect
from fastapi import status, Request
from fastapi.responses import JSONResponse
from cloudevents.http.conversion import from_http
from cloudevents.http.event import CloudEvent
from dapr_agents.agents.utils.text_printer import ColorTextFormatter
from dapr_agents.utils import add_signal_handlers_cross_platform
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from pydantic import BaseModel, Field, ValidationError, PrivateAttr
from dapr.clients import DaprClient
from dapr_agents.memory import (
    ConversationListMemory,
    ConversationVectorMemory,
    MemoryBase,
)
from dapr_agents.workflow.messaging import DaprPubSub
from dapr_agents.workflow.messaging.routing import MessageRoutingMixin
from dapr_agents.storage.daprstores.statestore import DaprStateStore
from dapr_agents.workflow import WorkflowApp

if TYPE_CHECKING:
    from fastapi import FastAPI

state_lock = threading.Lock()

logger = logging.getLogger(__name__)


class AgenticWorkflow(WorkflowApp, DaprPubSub, MessageRoutingMixin):
    """
    A class for managing agentic workflows, extending `WorkflowApp`.
    Handles agent interactions, workflow execution, messaging, and metadata management.
    """

    name: str = Field(..., description="The name of the agentic system.")
    message_bus_name: str = Field(
        ..., description="Dapr message bus component for pub/sub messaging."
    )
    broadcast_topic_name: Optional[str] = Field(
        "beacon_channel", description="Default topic for broadcasting messages."
    )
    state_store_name: str = Field(
        ..., description="Dapr state store for workflow state."
    )
    state_key: str = Field(
        default="workflow_state",
        description="Dapr state key for workflow state storage.",
    )
    state: Optional[Union[BaseModel, dict]] = Field(
        default=None, description="Current state of the workflow."
    )
    state_format: Optional[Type[BaseModel]] = Field(
        default=None, description="Schema to enforce state structure."
    )
    agents_registry_store_name: str = Field(
        ..., description="Dapr state store for agent metadata."
    )
    agents_registry_key: str = Field(
        default="agents_registry", description="Key for agents registry in state store."
    )
    # TODO: test this is respected by runtime.
    max_iterations: int = Field(
        default=10, description="Maximum iterations for workflows.", ge=1
    )
    memory: MemoryBase = Field(
        default_factory=ConversationListMemory,
        description="Handles conversation history storage.",
    )
    save_state_locally: bool = Field(
        default=True, description="Whether to save workflow state locally."
    )
    local_state_path: Optional[str] = Field(
        default=None, description="Local path for saving state files."
    )

    # Private internal attributes (not schema/validated)
    _state_store_client: Optional[DaprStateStore] = PrivateAttr(default=None)
    _text_formatter: [ColorTextFormatter] = PrivateAttr(default=ColorTextFormatter)
    _agent_metadata: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _workflow_name: str = PrivateAttr(default=None)
    _dapr_client: Optional[DaprClient] = PrivateAttr(default=None)
    _is_running: bool = PrivateAttr(default=False)
    _shutdown_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _http_server: Optional[Any] = PrivateAttr(default=None)
    _subscriptions: Dict[str, Callable] = PrivateAttr(default_factory=dict)
    _topic_handlers: Dict[
        Tuple[str, str], Dict[Type[BaseModel], Callable]
    ] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Initializes the workflow service, messaging, and metadata storage."""
        if not self._is_dapr_available():
            self._raise_dapr_required_error()

        # Set up color formatter for logging and CLI printing
        self._text_formatter = ColorTextFormatter()

        # Initialize state store client (used for persisting workflow state to Dapr)
        # Why make a state store client and dapr_client here?
        self._state_store_client = DaprStateStore(store_name=self.state_store_name)
        logger.info(f"State store '{self.state_store_name}' initialized.")

        # Load or initialize the current workflow state
        self.initialize_state()

        # Create a Dapr client for service-to-service calls or state interactions
        self._dapr_client = DaprClient()

        super().model_post_init(__context)

    @property
    def app(self) -> "FastAPI":
        """
        Returns the FastAPI application instance if the workflow was initialized as a service.

        Raises:
            RuntimeError: If the FastAPI server has not been initialized via `.as_service()` first.
        """
        if self._http_server:
            return self._http_server.app
        raise RuntimeError("FastAPI server not initialized. Call `as_service()` first.")

    def register_routes(self):
        """
        Hook method to register user-defined routes via `@route(...)` decorators,
        including FastAPI route options like `tags`, `summary`, `response_model`, etc.
        """
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if getattr(method, "_is_fastapi_route", False):
                path = getattr(method, "_route_path")
                method_type = getattr(method, "_route_method", "GET")
                extra_kwargs = getattr(method, "_route_kwargs", {})

                logger.info(f"Registering route {method_type} {path} -> {name}")
                self.app.add_api_route(
                    path, method, methods=[method_type], **extra_kwargs
                )

    def as_service(self, port: Optional[int] = None, host: str = "0.0.0.0"):
        """
        Enables FastAPI-based service mode for the agent by initializing a FastAPI server instance.
        Must be called before `start()` if you want to expose HTTP endpoints.

        Args:
            port: Required port number.
            host: Host address to bind to. Defaults to "0.0.0.0".

        Raises:
            ValueError: If port is not provided.
        """
        from dapr_agents.service.fastapi import FastAPIServerBase

        if port is None:
            raise ValueError("Port must be provided as a parameter")

        self._http_server = FastAPIServerBase(
            service_name=self.name,
            service_port=port,
            service_host=host,
        )

        # Register built-in routes
        self.app.add_api_route("/status", lambda: {"ok": True})
        self.app.add_api_route(
            "/start-workflow", self.run_workflow_from_request, methods=["POST"]
        )

        # Allow subclass to register additional routes
        self.register_routes()

        return self

    def handle_shutdown_signal(self, sig):
        logger.info(f"Shutdown signal {sig} received. Stopping service gracefully...")
        self._shutdown_event.set()
        asyncio.create_task(self.stop())

    async def start(self):
        """
        Starts the agent workflow service, optionally as a FastAPI server if .as_service() was called.
        Registers signal handlers and message routes if running in headless mode.
        """
        if self._is_running:
            logger.warning(
                "Service is already running. Ignoring duplicate start request."
            )
            return

        logger.info("Starting Agent Workflow Service...")
        self._shutdown_event.clear()

        try:
            # Headless mode (no HTTP server)
            if not hasattr(self, "_http_server") or self._http_server is None:
                logger.info("Running in headless mode.")

                # Add signal handlers using cross-platform approach for graceful shutdown
                loop = asyncio.get_event_loop()
                add_signal_handlers_cross_platform(loop, self.handle_shutdown_signal)

                self.register_message_routes()

                self._is_running = True
                while not self._shutdown_event.is_set():
                    await asyncio.sleep(1)

            # FastAPI mode
            else:
                logger.info("Running in FastAPI service mode.")
                self.register_message_routes()
                self._is_running = True
                await self._http_server.start()

        except asyncio.CancelledError:
            logger.info("Service received cancellation signal.")
        finally:
            await self.stop()

    async def stop(self):
        """
        Gracefully stops the agent service by unsubscribing and stopping the HTTP server if present.
        """
        if not self._is_running:
            logger.warning("Service is not running. Ignoring stop request.")
            return

        logger.info("Stopping Agent Workflow Service...")

        # Unsubscribe from all topics
        for (pubsub_name, topic_name), close_fn in self._subscriptions.items():
            try:
                logger.info(
                    f"Unsubscribing from pubsub '{pubsub_name}' topic '{topic_name}'"
                )
                close_fn()
            except Exception as e:
                logger.error(f"Failed to unsubscribe from topic '{topic_name}': {e}")

        self._subscriptions.clear()

        # If running as FastAPI, stop the HTTP server
        if hasattr(self, "_http_server") and self._http_server:
            logger.info("Stopping FastAPI server...")
            await self._http_server.stop()

        # Stop the workflow runtime
        if self.wf_runtime_is_running:
            logger.info("Shutting down workflow runtime.")
            self.stop_runtime()
            self.wf_runtime_is_running = False

        self._is_running = False
        logger.info("Agent Workflow Service stopped successfully.")

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
        chat_history_messages = [
            msg.model_dump() if isinstance(msg, BaseModel) else msg
            for msg in chat_history
        ]
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
                logger.debug(
                    "User provided a state as a Pydantic model. Converting to dict."
                )
                self.state = self.state.model_dump()

            if not isinstance(self.state, dict):
                raise TypeError(
                    f"Invalid state type: {type(self.state)}. Expected dict or Pydantic model."
                )

            logger.debug(f"Workflow state initialized with {len(self.state)} key(s).")
            self.save_state()

        except Exception as e:
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
                logger.warning(
                    "No schema (state_format) provided; returning state as-is."
                )
                return state_data

            logger.info("Validating workflow state against schema.")
            validated_state: BaseModel = self.state_format(
                **state_data
            )  # Validate with Pydantic
            return validated_state.model_dump()  # Convert validated model to dict

        except ValidationError as e:
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
            if (
                not self._state_store_client
                or not self.state_store_name
                or not self.state_key
            ):
                logger.error("State store is not configured. Cannot load state.")
                raise RuntimeError(
                    "State store is not configured. Please provide 'state_store_name' and 'state_key'."
                )

            # Avoid overwriting state if self.state is already set
            if self.state:
                logger.info(
                    "Using existing in-memory state. Skipping load from storage."
                )
                return self.state

            has_state, state_data = self._state_store_client.try_get_state(
                self.state_key
            )

            if has_state and state_data:
                logger.info(
                    f"Existing state found for key '{self.state_key}'. Validating it."
                )

                if not isinstance(state_data, dict):
                    raise TypeError(
                        f"Invalid state type retrieved: {type(state_data)}. Expected dict."
                    )

                return (
                    self.validate_state(state_data) if self.state_format else state_data
                )

            logger.info(
                f"No existing state found for key '{self.state_key}'. Initializing empty state."
            )
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

    def save_state_to_disk(
        self, state_data: str, filename: Optional[str] = None
    ) -> None:
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
            with tempfile.NamedTemporaryFile(
                "w", dir=save_directory, delete=False
            ) as tmp_file:
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
                            logger.warning(
                                "Existing state file is corrupt or empty. Overwriting."
                            )

                # Merge new state into existing state
                new_state = (
                    json.loads(state_data)
                    if isinstance(state_data, str)
                    else state_data
                )
                merged_state = {**existing_state, **new_state}  # Merge updates

                # Write merged state back to a new temp file
                with open(temp_path, "w", encoding="utf-8") as file:
                    json.dump(merged_state, file, indent=4)

                # Atomically replace the old state file
                os.replace(temp_path, file_path)

            logger.debug(f"Workflow state saved locally at '{file_path}'.")

        except Exception as e:
            logger.error(f"Failed to save workflow state to disk: {e}")
            raise RuntimeError(f"Error saving workflow state to disk: {e}")

    def save_state(
        self,
        state: Optional[Union[dict, BaseModel, str]] = None,
        force_reload: bool = False,
    ) -> None:
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
            if (
                not self._state_store_client
                or not self.state_store_name
                or not self.state_key
            ):
                logger.error("State store is not configured. Cannot save state.")
                raise RuntimeError(
                    "State store is not configured. Please provide 'state_store_name' and 'state_key'."
                )

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
                raise TypeError(
                    f"Invalid state type: {type(self.state)}. Expected dict, BaseModel, or JSON string."
                )

            # Save state in Dapr
            self._state_store_client.save_state(self.state_key, state_to_save)
            logger.debug(f"Successfully saved state for key '{self.state_key}'.")

            # Save state locally if enabled
            if self.save_state_locally:
                self.save_state_to_disk(state_data=state_to_save)

            # Reload state after saving if requested
            if force_reload:
                self.state = self.load_state()
                logger.debug(f"State reloaded after saving for key '{self.state_key}'.")

        except Exception as e:
            logger.error(f"Failed to save state for key '{self.state_key}': {e}")
            raise

    def get_agents_metadata(
        self, exclude_self: bool = True, exclude_orchestrator: bool = False
    ) -> dict:
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
            agents_metadata = (
                self.get_data_from_store(
                    self.agents_registry_store_name, self.agents_registry_key
                )
                or {}
            )

            if agents_metadata:
                logger.info(
                    f"Agents found in '{self.agents_registry_store_name}' for key '{self.agents_registry_key}'."
                )

                # Filter based on self-exclusion and orchestrator exclusion
                filtered_metadata = {
                    name: metadata
                    for name, metadata in agents_metadata.items()
                    if not (
                        exclude_self and name == self.name
                    )  # Exclude self if requested
                    and not (
                        exclude_orchestrator and metadata.get("orchestrator", False)
                    )  # Exclude orchestrators only if exclude_orchestrator=True
                }

                if not filtered_metadata:
                    logger.info("No other agents found after filtering.")

                return filtered_metadata

            logger.info(
                f"No agents found in '{self.agents_registry_store_name}' for key '{self.agents_registry_key}'."
            )
            return {}
        except Exception as e:
            logger.error(f"Failed to retrieve agents metadata: {e}", exc_info=True)
            raise RuntimeError(f"Error retrieving agents metadata: {str(e)}") from e

    async def broadcast_message(
        self,
        message: Union[BaseModel, dict],
        exclude_orchestrator: bool = False,
        **kwargs,
    ) -> None:
        """
        Sends a message to all agents (or only to non-orchestrator agents if exclude_orchestrator=True).

        Args:
            message (Union[BaseModel, dict]): The message content as a Pydantic model or dictionary.
            exclude_orchestrator (bool, optional): If True, excludes orchestrators from receiving the message. Defaults to False.
            **kwargs: Additional metadata fields to include in the message.
        """
        try:
            # Retrieve agents metadata while respecting the exclude_orchestrator flag
            agents_metadata = self.get_agents_metadata(
                exclude_orchestrator=exclude_orchestrator
            )

            if not agents_metadata:
                logger.warning("No agents available for broadcast.")
                return

            logger.info(
                f"{self.name} broadcasting message to {self.broadcast_topic_name}."
            )

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

    async def send_message_to_agent(
        self, name: str, message: Union[BaseModel, dict], **kwargs
    ) -> None:
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
                logger.warning(
                    f"Target '{name}' is not registered as an agent. Skipping message send."
                )
                return  # Do not raise an error—just warn and move on.

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
            logger.error(
                f"Failed to send message to agent '{name}': {e}", exc_info=True
            )

    def print_interaction(
        self, sender_agent_name: str, recipient_agent_name: str, message: str
    ) -> None:
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
        self._text_formatter.print_colored_text(interaction_text)

    def register_agentic_system(self) -> None:
        """
        Registers the agent's metadata in the Dapr state store under 'agents_metadata'.
        """
        try:
            # Update the agents registry store with the new agent metadata
            self.register_agent(
                store_name=self.agents_registry_store_name,
                store_key=self.agents_registry_key,
                agent_name=self.name,
                agent_metadata=self._agent_metadata,
            )
        except Exception as e:
            logger.error(f"Failed to register metadata for agent {self.name}: {e}")
            raise e

    async def run_workflow_from_request(self, request: Request) -> JSONResponse:
        """
        Run a workflow instance triggered by an incoming HTTP POST request.
        Supports dynamic workflow name via query param (?name=...).

        Args:
            request (Request): The incoming request containing input data for the workflow.

        Returns:
            JSONResponse: A 202 Accepted response with the workflow instance ID if successful,
                        or a 400/500 error response if the request fails validation or execution.
        """
        try:
            # Extract workflow name from query parameters or use default
            workflow_name = request.query_params.get("name") or self._workflow_name
            if not workflow_name:
                return JSONResponse(
                    content={"error": "No workflow name specified."},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            # Validate workflow name against registered workflows
            if workflow_name not in self.workflows:
                return JSONResponse(
                    content={
                        "error": f"Unknown workflow '{workflow_name}'. Available: {list(self.workflows.keys())}"
                    },
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            # Parse body as CloudEvent or fallback to JSON
            try:
                event: CloudEvent = from_http(
                    dict(request.headers), await request.body()
                )
                input_data = event.data
            except Exception:
                input_data = await request.json()

            logger.info(f"Starting workflow '{workflow_name}' with input: {input_data}")
            instance_id = self.run_workflow(workflow=workflow_name, input=input_data)

            asyncio.create_task(self.monitor_workflow_completion(instance_id))

            return JSONResponse(
                content={
                    "message": "Workflow initiated successfully.",
                    "workflow_instance_id": instance_id,
                },
                status_code=status.HTTP_202_ACCEPTED,
            )

        except Exception as e:
            logger.error(f"Error starting workflow: {str(e)}", exc_info=True)
            return JSONResponse(
                content={"error": "Failed to start workflow", "details": str(e)},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _is_dapr_available(self) -> bool:
        """
        Check if Dapr is available by attempting to connect to the Dapr sidecar.
        This provides better DX for users who don't have dapr running to see a nice error message.

        Returns:
            bool: True if Dapr is available, False otherwise
        """
        try:
            import socket
            import os

            def check_tcp_port(port: int, timeout: int = 2) -> bool:
                """Check if a TCP port is open and accepting connections."""
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(timeout)
                    result = sock.connect_ex(("localhost", port))
                    sock.close()
                    return result == 0
                except Exception:
                    return False

            ports_to_check = []
            for env_var in ["DAPR_HTTP_PORT", "DAPR_GRPC_PORT"]:
                port = os.environ.get(env_var)
                if port:
                    ports_to_check.append(int(port))

            # Fallback ports
            ports_to_check.extend([3500, 3501, 3502])
            for port in ports_to_check:
                if check_tcp_port(port):
                    return True

            return False
        except Exception:
            return False

    def _raise_dapr_required_error(self):
        """
        Raise a helpful error message when Dapr is required but not available.
        """
        error_msg = """🚫 Dapr Required for Durable Agent

This agent requires Dapr to be running because it uses stateful, durable workflows.

To run this agent, you need to:

1. Install Dapr CLI: https://docs.dapr.io/getting-started/install-dapr-cli/
2. Initialize Dapr: dapr init
3. Run with Dapr: dapr run --app-id your-app-id --app-port 8001 --resources-path components/ -- python your_script.py

For more information, see the README.md in the quickstart directory."""
        raise RuntimeError(error_msg)
