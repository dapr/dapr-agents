import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type, List

from cloudevents.http.conversion import from_http
from cloudevents.http.event import CloudEvent
from dapr.clients import DaprClient
from dapr.clients.grpc._request import (
    TransactionalStateOperation,
    TransactionOperationType,
)
from dapr.clients.grpc._response import StateResponse
from dapr.clients.grpc._state import Concurrency, Consistency, StateOptions
from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, PrivateAttr

from dapr_agents.agents.utils.text_printer import ColorTextFormatter
from dapr_agents.memory import (
    ConversationListMemory,
    ConversationVectorMemory,
    MemoryBase,
)
from dapr_agents.storage.daprstores.statestore import DaprStateStore
from dapr_agents.workflow.base import WorkflowApp
from dapr_agents.workflow.mixins import (
    MessagingMixin,
    PubSubMixin,
    ServiceMixin,
    StateManagementMixin,
)

logger = logging.getLogger(__name__)


class AgenticWorkflow(
    WorkflowApp,
    PubSubMixin,
    StateManagementMixin,
    ServiceMixin,
    MessagingMixin,
):
    """
    A class for managing agentic workflows, extending `WorkflowApp`.
    Handles agent interactions, workflow execution, messaging, and metadata management.
    """

    name: str = Field(..., description="The name of the agentic system.")
    message_bus_name: str = Field(
        ...,
        description="The name of the message bus component, defining the pub/sub base.",
    )
    broadcast_topic_name: Optional[str] = Field(
        default=None,
        description="Default topic for broadcasting messages. Set explicitly for multi-agent setups.",
    )
    state_store_name: str = Field(
        ..., description="Dapr state store for workflow state."
    )
    state_key: str = Field(
        default="workflow_state",
        description="Dapr state key for workflow state storage.",
    )
    state: dict = Field(
        default_factory=dict, description="Current state of the workflow."
    )
    state_format: Optional[Type[BaseModel]] = Field(
        default=None,
        description=(
            "Optional Pydantic model used to validate the persisted workflow "
            "state. If provided, state loaded from storage is coerced to this "
            "schema."
        ),
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
    client: Optional[DaprClient] = Field(
        default=None, init=False, description="Dapr client instance."
    )

    # Private internal attributes (not schema/validated)
    _state_store_client: Optional[DaprStateStore] = PrivateAttr(default=None)
    _text_formatter: ColorTextFormatter = PrivateAttr(default=ColorTextFormatter)
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
        """
        Post-initialization hook for the AgenticWorkflow.

        This method initializes the workflow service, messaging, and metadata storage.
        It sets up the color formatter, state store client, loads or initializes the workflow state,
        and creates a Dapr client for service-to-service calls or state interactions.

        Args:
            __context (Any): The context passed by Pydantic's model initialization.

        Raises:
            RuntimeError: If Dapr is not available in the current environment.
        """
        self._dapr_client = DaprClient()
        self._text_formatter = ColorTextFormatter()
        self._state_store_client = DaprStateStore(store_name=self.state_store_name)
        logger.info(f"State store '{self.state_store_name}' initialized.")
        self.initialize_state()
        super().model_post_init(__context)

    def get_chat_history(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the chat history from memory as a list of dictionaries.

        Args:
            task (Optional[str]): The task or query provided by the user (used for vector search).

        Returns:
            List[Dict[str, Any]]: The chat history as dictionaries.
        """
        if isinstance(self.memory, ConversationVectorMemory) and task:
            if (
                hasattr(self.memory.vector_store, "embedding_function")
                and self.memory.vector_store.embedding_function
                and hasattr(
                    self.memory.vector_store.embedding_function, "embed_documents"
                )
            ):
                query_embeddings = self.memory.vector_store.embedding_function.embed(
                    task
                )
                messages = self.memory.get_messages(query_embeddings=query_embeddings)
            else:
                messages = self.memory.get_messages()
        else:
            messages = self.memory.get_messages()
        return messages

    @property
    def chat_history(self) -> List[Dict[str, Any]]:
        """
        Returns the full chat history as a list of dictionaries.

        Returns:
            List[Dict[str, Any]]: The chat history.
        """
        return self.get_chat_history()

    def get_data_from_store(self, store_name: str, key: str) -> Optional[dict]:
        """
        Retrieves data from the Dapr state store using the given key.

        Args:
            store_name (str): The name of the Dapr state store component.
            key (str): The key to fetch data from.

        Returns:
            Optional[dict]: the retrieved dictionary or None if not found.
        """
        try:
            response: StateResponse = self._dapr_client.get_state(
                store_name=store_name, key=key
            )
            data = response.data
            return json.loads(data) if data else None
        except Exception:
            logger.warning(
                f"Error retrieving data for key '{key}' from store '{store_name}'"
            )
            return None

    def get_agents_metadata(
        self, exclude_self: bool = True, exclude_orchestrator: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve metadata for all registered agents.

        Args:
            exclude_self: If True, excludes the current agent from results.
            exclude_orchestrator: If True, excludes orchestrators from results.

        Returns:
            Dict[str, Any]: Mapping of agent names to their metadata.

        Raises:
            RuntimeError: If retrieval fails.
        """
        try:
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
                filtered = {
                    name: metadata
                    for name, metadata in agents_metadata.items()
                    if not (exclude_self and name == self.name)
                    and not (
                        exclude_orchestrator and metadata.get("orchestrator", False)
                    )
                }
                if not filtered:
                    logger.info("No other agents found after filtering.")
                return filtered

            logger.info(
                f"No agents found in '{self.agents_registry_store_name}' for key '{self.agents_registry_key}'."
            )
            return {}
        except Exception as e:
            logger.error(f"Failed to retrieve agents metadata: {e}", exc_info=True)
            raise RuntimeError(f"Error retrieving agents metadata: {str(e)}") from e

    def print_interaction(
        self, sender_agent_name: str, recipient_agent_name: str, message: str
    ) -> None:
        """
        Pretty-print an interaction between two agents.

        Args:
            sender_agent_name: The name of the agent sending the message.
            recipient_agent_name: The name of the agent receiving the message.
            message: The message content to display.
        """
        separator = "-" * 80
        interaction_text = [
            (sender_agent_name, "dapr_agents_mustard"),
            (" -> ", "dapr_agents_teal"),
            (f"{recipient_agent_name}\n\n", "dapr_agents_mustard"),
            (message + "\n\n", None),
            (separator + "\n", "dapr_agents_teal"),
        ]
        self._text_formatter.print_colored_text(interaction_text)

    def register_agent(
        self, store_name: str, store_key: str, agent_name: str, agent_metadata: dict
    ) -> None:
        """
        Merges the existing data with the new data and updates the store.

        Args:
            store_name (str): The name of the Dapr state store component.
            key (str): The key to update.
            data (dict): The data to update the store with.
        """
        # retry the entire operation up to twenty times sleeping 1-2 seconds between each
        # TODO: rm the custom retry logic here and use the DaprClient retry_policy instead.
        for attempt in range(1, 21):
            try:
                response: StateResponse = self._dapr_client.get_state(
                    store_name=store_name, key=store_key
                )
                if not response.etag:
                    # if there is no etag the following transaction won't work as expected
                    # so we need to save an empty object with a strong consistency to force the etag to be created
                    self._dapr_client.save_state(
                        store_name=store_name,
                        key=store_key,
                        value=json.dumps({}),
                        state_metadata={
                            "contentType": "application/json",
                            "partitionKey": store_key,
                        },
                        options=StateOptions(
                            concurrency=Concurrency.first_write,
                            consistency=Consistency.strong,
                        ),
                    )
                    # raise an exception to retry the entire operation
                    raise Exception(f"No etag found for key: {store_key}")
                existing_data = json.loads(response.data) if response.data else {}
                if (agent_name, agent_metadata) in existing_data.items():
                    logger.debug(f"agent {agent_name} already registered.")
                    return None
                agent_data = {agent_name: agent_metadata}
                merged_data = {**existing_data, **agent_data}
                logger.debug(f"merged data: {merged_data} etag: {response.etag}")
                try:
                    # using the transactional API to be able to later support the Dapr outbox pattern
                    self._dapr_client.execute_state_transaction(
                        store_name=store_name,
                        operations=[
                            TransactionalStateOperation(
                                key=store_key,
                                data=json.dumps(merged_data),
                                etag=response.etag,
                                operation_type=TransactionOperationType.upsert,
                            )
                        ],
                        transactional_metadata={
                            "contentType": "application/json",
                            "partitionKey": store_key,
                        },
                    )
                except Exception as e:
                    raise e
                return None
            except Exception as e:
                logger.error(f"Error on transaction attempt: {attempt}: {e}")
                # Add random jitter
                import random

                delay = 1 + random.uniform(0, 1)  # 1-2 seconds
                logger.info(
                    f"Sleeping for {delay:.2f} seconds before retrying transaction..."
                )
                time.sleep(delay)
        raise Exception(
            f"Failed to update state store key: {store_key} after 20 attempts."
        )

    def register_agentic_system(self) -> None:
        """
        Register this agent's metadata in the Dapr state store.

        Raises:
            Exception: If registration fails.
        """
        try:
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
        Run a workflow instance triggered by an HTTP POST request.

        Args:
            request: The incoming FastAPI request.

        Returns:
            JSONResponse: HTTP response with workflow instance ID or error.
        """
        try:
            workflow_name = request.query_params.get("name") or self._workflow_name
            if not workflow_name:
                return JSONResponse(
                    content={"error": "No workflow name specified."},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            if workflow_name not in self.workflows:
                return JSONResponse(
                    content={
                        "error": f"Unknown workflow '{workflow_name}'. Available: {list(self.workflows.keys())}"
                    },
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            try:
                event: CloudEvent = from_http(
                    dict(request.headers), await request.body()
                )
                input_data = event.data
            except Exception:
                input_data = await request.json()

            logger.info(f"Starting workflow '{workflow_name}' with input: {input_data}")
            instance_id = await self.run_and_monitor_workflow_async(
                workflow=workflow_name, input=input_data
            )

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
