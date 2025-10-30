import json
from pydantic import Field, BaseModel, PrivateAttr
from typing import Optional, Type, Dict, Any, List, Union
from datetime import datetime
import uuid
from dapr_agents.types import MessageContent, ToolExecutionRecord
from dapr_agents.types.workflow import DaprWorkflowStatus
import logging
from dapr_agents.types import BaseMessage
from dapr.clients import DaprClient

logger = logging.getLogger(__name__)


class MemoryStore(BaseModel):
    """
    Unified storage for both Agent and DurableAgent.

    For regular Agent:
    - If `name` is None: Pure in-memory operation (no persistence, no registration)
    - If `name` is provided:
        - Conversation history: Persistent in Dapr state store

    For DurableAgent:
    - Requires `name` to be provided
    - Conversation history: Persistent in Dapr state store (via workflow instances)
    - Workflow state: Full workflow instance tracking with sessions
    """

    name: Optional[str] = Field(
        default=None,
        description=(
            "Dapr state store name. "
            "For Agent: If set, stores conversation and registers agent. If None, pure in-memory. "
            "For DurableAgent: Required. Stores workflow state, conversation, and registers agent."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session ID to group related conversations and workflow state. "
            "If not provided, a unique ID will be generated. "
            "Can be overridden in agent.run()."
        ),
    )
    local_directory: Optional[str] = Field(
        default=None,
        description=(
            "Directory path where state files will be saved locally. "
            "If not set, no local saving occurs. "
            "Can be absolute or relative to workspace root. "
            "Files will be named '{agent_name}_state.json'."
        ),
    )
    # TODO: remove this when we remove state mixin class and just use db state...
    _current_state: dict = PrivateAttr(default_factory=dict)
    _agent_name: str = PrivateAttr(default=None)  # Set by AgenticWorkflow
    _key: str = PrivateAttr(default="workflow_state")
    _in_memory_messages: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _dapr_client: DaprClient = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.name is None:
            if self._dapr_client is not None:
                logger.warning(
                    "DaprClient initialized but name is None. It will be ignored."
                )
            self._dapr_client = None
        else:
            self._dapr_client = DaprClient()

    def _set_key(self, agent_name: str) -> None:
        """Internal method to set the agent name and initialize storage."""
        self._agent_name = agent_name
        self._key = f"{agent_name}_workflow_state"

    def _get_instance_key(self, instance_id: str) -> str:
        """Get the state store key for a workflow instance."""
        return f"{self._agent_name}_workflow_{instance_id}"

    def _get_session_key(self, session_id: str) -> str:
        """Get the state store key for a specific session."""
        return f"{self._agent_name}_session_{session_id}"

    def _get_sessions_index_key(self) -> str:
        """Get the state store key for the sessions index."""
        return f"{self._agent_name}_sessions"

    def _get_session_id(self) -> str:
        """Get or generate a session ID."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        return self.session_id

    def _update_session_index(self, instance_id: str) -> None:
        """Update session index with workflow instance."""
        if not self.name:
            logger.debug("In-memory mode: skipping session update")
            return

        session_id = self._get_session_id()
        session_key = self._get_session_key(session_id)
        sessions_index_key = self._get_sessions_index_key()

        response = self._dapr_client.get_state(self.name, session_key)

        session_data = {}
        is_new_session = not bool(response.data)

        if response.data:
            # Safely decode and parse
            raw = response.data
            if isinstance(raw, (bytes, bytearray)):
                try:
                    raw = raw.decode("utf-8")
                except UnicodeDecodeError:
                    logger.error(f"Failed to decode session data for '{session_key}'")
                    raw = ""
            if isinstance(raw, str) and raw.strip():
                try:
                    session_data = json.loads(raw)
                    if not isinstance(session_data, dict):
                        logger.warning(
                            f"Session data not a dict, resetting: {type(session_data)}"
                        )
                        session_data = {}
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid session JSON for '{session_key}': {e}")
                    session_data = {}
            else:
                session_data = {}

        if is_new_session:
            session_data = {
                "session_id": session_id,
                "workflow_instances": [],
                "metadata": {
                    "agent_name": self._agent_name,
                    "created_at": datetime.now().isoformat(),
                },
                "last_active": datetime.now().isoformat(),
            }
            logger.debug(f"Created new session '{session_id}'")

        instances = session_data.get("workflow_instances", [])
        if instance_id not in instances:
            instances.append(instance_id)
            session_data["workflow_instances"] = instances
            session_data["last_active"] = datetime.now().isoformat()
            logger.debug(f"Added instance '{instance_id}' to session '{session_id}'")

        # === 4. Save session ===
        self._save_state_with_metadata(session_key, session_data)

        # update sessions index - only on first instance
        if is_new_session:
            index_resp = self._dapr_client.get_state(self.name, sessions_index_key)
            index_data = {"sessions": [], "last_updated": datetime.now().isoformat()}

            if index_resp.data:
                raw = index_resp.data
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                if raw.strip():
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            index_data["sessions"] = parsed.get("sessions", [])
                            index_data["last_updated"] = parsed.get(
                                "last_updated", index_data["last_updated"]
                            )
                    except json.JSONDecodeError:
                        logger.warning("Corrupted sessions index, resetting")

            if session_id not in index_data["sessions"]:
                index_data["sessions"].append(session_id)
                index_data["last_updated"] = datetime.now().isoformat()
                self._save_state_with_metadata(sessions_index_key, index_data)
                logger.debug(f"Registered session '{session_id}' in index")

    # TODO: in future remove this in favor of just using client.save_state when we use objects and not dictionaries in storage.
    def _save_state_with_metadata(self, key: str, data: Any) -> None:
        """Save state with content type metadata."""
        # Serialize data to JSON string if it's not already
        if isinstance(data, dict):
            data_to_save = json.dumps(data)
        elif isinstance(data, str):
            data_to_save = data
        else:
            data_to_save = json.dumps(data)

        self._dapr_client.save_state(
            self.name,
            key,
            data_to_save,
            state_metadata={"contentType": "application/json"},
        )

    def is_persistent(self) -> bool:
        """Check if storage is persistent (has a state store name) or in-memory."""
        return self.name is not None

    def add_message(self, message: Union[Dict[str, Any], "BaseMessage"]) -> None:
        """
        Add a single message to storage.
        - If name is None: Uses in-memory list
        - If name is set: Stores in Dapr state store

        Args:
            message (Union[Dict[str, Any], BaseMessage]): The message to add
        """
        msg_dict = message.model_dump() if hasattr(message, "model_dump") else message

        if self.is_persistent():
            # Save to state store
            messages = self.get_messages()
            messages.append(msg_dict)
            self._save_messages_to_store(messages)
        else:
            # In-memory mode
            self._in_memory_messages.append(msg_dict)

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """
        Add multiple messages to storage.
        - If name is None: Uses in-memory list
        - If name is set: Stores in Dapr state store

        Args:
            messages (List[Dict[str, Any]]): The messages to add
        """
        if self.is_persistent():
            # Save to state store
            current_messages = self.get_messages()
            current_messages.extend(messages)
            self._save_messages_to_store(current_messages)
        else:
            # In-memory mode
            self._in_memory_messages.extend(messages)

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages from storage.
        - If name is None: Returns in-memory list
        - If name is set: Loads from Dapr state store

        Returns:
            List[Dict[str, Any]]: All stored messages
        """
        if self.is_persistent():
            # Load from state store
            return self._load_messages_from_store()
        else:
            # In-memory mode
            return self._in_memory_messages.copy()

    def _get_messages_key(self) -> str:
        """Get the state store key for conversation messages."""
        session_id = self._get_session_id()
        return f"{self._agent_name}_messages_{session_id}"

    def _save_messages_to_store(self, messages: List[Dict[str, Any]]) -> None:
        """Save messages to the Dapr state store."""
        key = self._get_messages_key()
        data = json.dumps({"messages": messages})

        self._dapr_client.save_state(
            store_name=self.name,
            key=key,
            value=data,
            state_metadata={"contentType": "application/json"},
        )

    def _load_messages_from_store(self) -> List[Dict[str, Any]]:
        """Load messages from the Dapr state store."""
        key = self._get_messages_key()
        response = self._dapr_client.get_state(store_name=self.name, key=key)
        if response.data:
            data = (
                json.loads(response.data)
                if isinstance(response.data, (str, bytes))
                else response.data
            )
            return data.get("messages", [])
        return []


class DurableAgentMessage(MessageContent):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the message was created",
    )


class DurableAgentWorkflowEntry(BaseModel):
    """Represents a workflow and its associated data, including metadata on the source of the task request."""

    input: str = Field(
        ..., description="The input or description of the Workflow to be performed"
    )
    output: Optional[str] = Field(
        default=None, description="The output or result of the Workflow, if completed"
    )
    start_time: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the workflow was started",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the workflow was completed or failed",
    )
    messages: List[DurableAgentMessage] = Field(
        default_factory=list,
        description="Messages exchanged during the workflow (user, assistant, or tool messages).",
    )
    last_message: Optional[DurableAgentMessage] = Field(
        default=None, description="Last processed message in the workflow"
    )
    tool_history: List[ToolExecutionRecord] = Field(
        default_factory=list, description="Tool message exchanged during the workflow"
    )
    source: Optional[str] = Field(None, description="Entity that initiated the task.")
    workflow_instance_id: Optional[str] = Field(
        default=None,
        description="The agent's own workflow instance ID.",
    )
    triggering_workflow_instance_id: Optional[str] = Field(
        default=None,
        description="The workflow instance ID of the entity that triggered this agent (for multi-agent communication).",
    )
    workflow_name: Optional[str] = Field(
        default=None,
        description="The name of the workflow.",
    )
    trace_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="OpenTelemetry trace context for workflow resumption.",
    )
    status: str = Field(
        default=DaprWorkflowStatus.RUNNING.value,
        description="Current status of the workflow.",
    )


class DurableAgentWorkflowState(BaseModel):
    """Represents the state of multiple Agent workflows."""

    instances: Dict[str, DurableAgentWorkflowEntry] = Field(
        default_factory=dict,
        description="Workflow entries indexed by their instance_id.",
    )
    chat_history: List[DurableAgentMessage] = Field(
        default_factory=list,
        description="Chat history of messages exchanged during the workflow.",
    )
