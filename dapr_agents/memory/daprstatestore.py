import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from dapr_agents.memory import MemoryBase
from dapr_agents.storage.daprstores.statestore import DaprStateStore
from dapr_agents.types import BaseMessage

logger = logging.getLogger(__name__)


class ConversationDaprStateMemory(MemoryBase):
    """
    Manages conversation memory stored in a Dapr state store. Each message in the conversation is saved
    individually with a unique key and includes a workflow instance ID and timestamp for querying and retrieval.
    Key format: {agent_name}:agent_memory_{workflow_instance_id}
    """

    store_name: str = Field(
        default="statestore", description="The name of the Dapr state store."
    )
    agent_name: str = Field(
        default="default",
        description="Agent name for key namespacing; set by the framework in init.",
    )

    dapr_store: Optional[DaprStateStore] = Field(
        default=None, init=False, description="Dapr State Store."
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the Dapr state store after validation.
        """
        self.dapr_store = DaprStateStore(store_name=self.store_name)
        logger.info(
            "ConversationDaprStateMemory initialized (store=%s, agent_name=%s)",
            self.store_name,
            self.agent_name,
        )
        super().model_post_init(__context)

    def _get_message_key(self, workflow_instance_id: str, message_id: str) -> str:
        """Composite key for a message: agent_name:agent_memory_{workflow_instance_id}:{message_id}."""
        return f"{self._memory_key(workflow_instance_id)}:{message_id}"

    def _memory_key(self, workflow_instance_id: str) -> str:
        """Build state store key: agent_name:agent_memory_{workflow_instance_id}."""
        normalized = self.agent_name.replace(" ", "-").lower()
        return f"{normalized}:_memory_{workflow_instance_id}"

    def add_message(
        self,
        message: Union[Dict[str, Any], BaseMessage],
        workflow_instance_id: str,
    ) -> None:
        """
        Adds a single message to the memory and saves it to the Dapr state store.

        Args:
            message: The message to add to the memory.
            workflow_instance_id: Workflow instance id to add message for.
        """
        key = self._memory_key(workflow_instance_id)
        message = self._convert_to_dict(message)
        message.update(
            {
                "createdAt": datetime.now().isoformat() + "Z",
            }
        )

        # Retry loop for optimistic concurrency control
        # TODO: make this nicer in future, but for durability this must all be atomic
        max_attempts = 10
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.dapr_store.get_state(
                    key,
                    state_metadata={"contentType": "application/json"},
                )

                if response and response.data:
                    existing = json.loads(response.data)
                    etag = response.etag
                else:
                    existing = []
                    etag = None

                existing.append(message)
                # Save with etag - will fail if someone else modified it
                self.dapr_store.save_state(
                    key,
                    json.dumps(existing),
                    state_metadata={"contentType": "application/json"},
                    etag=etag,
                )

                # Success - exit retry loop
                return

            except Exception as exc:
                if attempt == max_attempts:
                    logger.exception(
                        "Failed to add message to workflow instance %s after %s attempts: %s",
                        key,
                        max_attempts,
                        exc,
                    )
                    raise
                else:
                    logger.warning(
                        "Conflict adding message to workflow instance %s (attempt %s/%s): %s, retrying...",
                        key,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    # Brief exponential backoff with jitter
                    import time
                    import random

                    time.sleep(min(0.1 * attempt, 0.5) * (1 + random.uniform(0, 0.25)))

    def add_messages(
        self,
        messages: List[Union[Dict[str, Any], BaseMessage]],
        workflow_instance_id: str,
    ) -> None:
        """
        Adds multiple messages to the memory and saves each one to the Dapr state store.

        Args:
            messages: A list of messages to add to the memory.
            workflow_instance_id: Workflow instance id for these messages.
        """
        logger.info(
            "Adding %s messages to workflow instance %s",
            len(messages),
            workflow_instance_id,
        )
        for message in messages:
            self.add_message(message, workflow_instance_id)

    def add_interaction(
        self,
        user_message: Union[Dict[str, Any], BaseMessage],
        assistant_message: Union[Dict[str, Any], BaseMessage],
        workflow_instance_id: str,
    ) -> None:
        """
        Adds a user-assistant interaction to the memory storage and saves it to the state store.

        Args:
            user_message: The user message.
            assistant_message: The assistant message.
            workflow_instance_id: Workflow instance id for this interaction.
        """
        self.add_messages([user_message, assistant_message], workflow_instance_id)

    def _decode_message(self, message_data: Union[bytes, str]) -> Dict[str, Any]:
        """
        Decodes the message data if it's in bytes, otherwise parses it as a JSON string.

        Args:
            message_data (Union[bytes, str]): The message data to decode.

        Returns:
            Dict[str, Any]: The decoded message as a dictionary.
        """
        if isinstance(message_data, bytes):
            message_data = message_data.decode("utf-8")
        return json.loads(message_data)

    def get_messages(
        self, workflow_instance_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieves messages stored in the state store for the given workflow instance.

        Args:
            workflow_instance_id: Workflow instance id to retrieve messages for.
            limit: Maximum number of messages to retrieve. Defaults to 100.

        Returns:
            List of message dicts with all fields.
        """
        response = self.query_messages(workflow_instance_id)
        if response and hasattr(response, "data") and response.data:
            raw_messages = json.loads(response.data)
            if raw_messages:
                messages = raw_messages[:limit]
                logger.info(
                    "Retrieved %s messages for workflow instance %s",
                    len(messages),
                    workflow_instance_id,
                )
                return messages
        return []

    def query_messages(self, workflow_instance_id: str) -> Any:
        """
        Queries messages from the state store for the given workflow instance.

        Args:
            workflow_instance_id: Workflow instance id to query.

        Returns:
            Response object from the Dapr state store with 'data' containing messages as JSON.
        """
        key = self._memory_key(workflow_instance_id)
        logger.debug("Executing query for workflow instance %s", workflow_instance_id)
        states_metadata = {"contentType": "application/json"}
        response = self.dapr_store.get_state(
            key, state_metadata=states_metadata
        )
        return response

    def reset_memory(self, workflow_instance_id: str) -> None:
        """
        Clears all messages stored in the memory for the given workflow instance.

        Args:
            workflow_instance_id: Workflow instance id to reset.
        """
        key = self._memory_key(workflow_instance_id)
        self.dapr_store.delete_state(key)
        logger.info(
            "Memory reset for workflow instance %s completed.", workflow_instance_id
        )
