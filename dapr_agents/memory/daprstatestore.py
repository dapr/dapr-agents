from dapr_agents.storage.daprstores.statestore import DaprStateStore
from dapr_agents.types import BaseMessage
from dapr_agents.memory import MemoryBase
from typing import List, Union, Optional, Dict, Any
from pydantic import Field, model_validator
from datetime import datetime
import json
import uuid
import logging

logger = logging.getLogger(__name__)


def generate_numeric_session_id() -> int:
    """
    Generates a random numeric session ID by extracting digits from a UUID.

    Returns:
        int: A numeric session ID.
    """
    return int("".join(filter(str.isdigit, str(uuid.uuid4()))))


class ConversationDaprStateMemory(MemoryBase):
    """
    Manages conversation memory stored in a Dapr state store. Each message in the conversation is saved
    individually with a unique key and includes a session ID and timestamp for querying and retrieval.
    """

    store_name: str = Field(
        default="statestore", description="The name of the Dapr state store."
    )
    session_id: Optional[Union[str, int]] = Field(
        default=None, description="Unique identifier for the conversation session."
    )

    # Private attribute to hold the initialized DaprStateStore
    dapr_store: Optional[DaprStateStore] = Field(
        default=None, init=False, description="Dapr State Store."
    )

    @model_validator(mode="before")
    def set_session_id(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sets a numeric session ID if none is provided.

        Args:
            values (Dict[str, Any]): The dictionary of attribute values before initialization.

        Returns:
            Dict[str, Any]: Updated values including the generated session ID if not provided.
        """
        if not values.get("session_id"):
            values["session_id"] = generate_numeric_session_id()
        return values

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the Dapr state store after validation
        """
        self.dapr_store = DaprStateStore(store_name=self.store_name)
        logger.info(
            f"ConversationDaprStateMemory initialized with session ID: {self.session_id}"
        )

        # Complete post-initialization
        super().model_post_init(__context)

    def _get_message_key(self, message_id: str) -> str:
        """
        Generates a unique key for each message using session_id and message_id.

        Args:
            message_id (str): A unique identifier for the message.

        Returns:
            str: A composite key for storing individual messages.
        """
        return f"{self.session_id}:{message_id}"

    def add_message(self, message: Union[Dict, BaseMessage]):
        """
        Adds a single message to the memory and saves it to the Dapr state store.

        Args:
            message (Union[Dict, BaseMessage]): The message to add to the memory.
        """

        if isinstance(message, BaseMessage):
            message = message.model_dump()

        message_id = str(uuid.uuid4())
        message_key = self._get_message_key(message_id)
        message.update(
            {
                "sessionId": self.session_id,
                "createdAt": datetime.now().isoformat() + "Z",
            }
        )

        existing = self.get_messages()
        existing.append(message)

        logger.debug(
            f"Adding message with key {message_key} to session {self.session_id}"
        )
        self.dapr_store.save_state(
            self.session_id, json.dumps(existing), {"contentType": "application/json"}
        )

    def add_messages(self, messages: List[Union[Dict, BaseMessage]]):
        """
        Adds multiple messages to the memory and saves each one individually to the Dapr state store.

        Args:
            messages (List[Union[Dict, BaseMessage]]): A list of messages to add to the memory.
        """
        logger.info(f"Adding {len(messages)} messages to session {self.session_id}")
        for message in messages:
            if isinstance(message, BaseMessage):
                message = message.model_dump()
            self.add_message(message)

    def add_interaction(
        self, user_message: BaseMessage, assistant_message: BaseMessage
    ):
        """
        Adds a user-assistant interaction to the memory storage and saves it to the state store.

        Args:
            user_message (BaseMessage): The user message.
            assistant_message (BaseMessage): The assistant message.
        """
        self.add_messages([user_message, assistant_message])

    def _decode_message(self, message_data: Union[bytes, str]) -> dict:
        """
        Decodes the message data if it's in bytes, otherwise parses it as a JSON string.

        Args:
            message_data (Union[bytes, str]): The message data to decode.

        Returns:
            dict: The decoded message as a dictionary.
        """
        if isinstance(message_data, bytes):
            message_data = message_data.decode("utf-8")
        return json.loads(message_data)

    def get_messages(self, limit: int = 100) -> List[Dict[str, str]]:
        """
        Retrieves messages stored in the state store for the current session_id, with an optional limit.

        Args:
            limit (int): The maximum number of messages to retrieve. Defaults to 100.

        Returns:
            List[Dict[str, str]]: A list containing the 'content' and 'role' fields of the messages.
        """
        response = self.query_messages(session_id=self.session_id)
        if response and response.data:
            raw_messages = json.loads(response.data)
            if raw_messages:
                messages = [
                    {"content": msg.get("content"), "role": msg.get("role")}
                    for msg in raw_messages
                ]

                logger.info(
                    f"Retrieved {len(messages)} messages for session {self.session_id}"
                )
                return messages

        return []

    def query_messages(self, session_id: str) -> List[Dict[str, str]]:
        """
        Queries messages from the state store based on a pre-constructed query string.

        Args:
            query (Optional[str]): A JSON-formatted query string to be executed.

        Returns:
            List[Dict[str, str]]: A list containing the 'content' and 'role' fields of the messages.
        """
        logger.debug(f"Executing query for session {self.session_id}")
        states_metadata = {"contentType": "application/json"}
        response = self.dapr_store.get_state(session_id, state_metadata=states_metadata)
        return response

    def reset_memory(self):
        """
        Clears all messages stored in the memory and resets the state store for the current session.
        """
        self.dapr_store.delete_state(self.session_id)
        logger.info(f"Memory reset for session {self.session_id} completed.")
