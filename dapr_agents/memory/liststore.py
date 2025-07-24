from typing import Any, Dict, List, Union

from pydantic import Field

from dapr_agents.memory import MemoryBase
from dapr_agents.types import BaseMessage


class ConversationListMemory(MemoryBase):
    """
    Memory storage for conversation messages using a list-based approach. This class provides a simple way to store,
    retrieve, and manage messages during a conversation session.
    """

    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of messages stored in conversation memory as dictionaries.",
    )

    def add_message(self, message: Union[Dict[str, Any], BaseMessage]):
        """
        Adds a single message to the end of the memory list.

        Args:
            message (Union[Dict, BaseMessage]): The message to add to the memory.
        """
        self.messages.append(self._convert_to_dict(message))

    def add_messages(self, messages: List[Union[Dict[str, Any], BaseMessage]]):
        """
        Adds multiple messages to the memory by appending each message from the provided list to the end of the memory list.

        Args:
            messages (List[Union[Dict, BaseMessage]]): A list of messages to add to the memory.
        """
        self.messages.extend(self._convert_to_dict(msg) for msg in messages)

    def add_interaction(
        self, user_message: BaseMessage, assistant_message: BaseMessage
    ):
        """
        Adds a user-assistant interaction to the memory storage.

        Args:
            user_message (BaseMessage): The user message.
            assistant_message (BaseMessage): The assistant message.
        """
        self.add_messages([user_message, assistant_message])

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Retrieves a copy of all messages stored in the memory.

        Returns:
            List[Dict[str, Any]]: A list containing copies of all stored messages as dictionaries.
        """
        return self.messages.copy()

    def reset_memory(self):
        """Clears all messages stored in the memory, resetting the memory to an empty state."""
        self.messages.clear()
