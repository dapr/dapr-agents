from abc import ABC, abstractmethod
from typing import Dict, List, Union, Any

from pydantic import BaseModel, ConfigDict

from dapr_agents.types import BaseMessage


class MemoryBase(BaseModel, ABC):
    """
    Abstract base class for managing message memory. This class defines a standard interface for memory operations,
    allowing for different implementations of message storage mechanisms in subclasses.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def add_message(self, message: BaseMessage, workflow_instance_id: str):
        """
        Adds a single message to the memory storage.

        Args:
            message (BaseMessage): The message object to be added.
            workflow_instance_id: Workflow instance id for this message.

        Note:
            This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def add_messages(self, messages: List[BaseMessage], workflow_instance_id: str):
        """
        Adds a list of messages to the memory storage.

        Args:
            messages (List[BaseMessage]): A list of message objects to be added.
            workflow_instance_id: Workflow instance id for these messages.

        Note:
            This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def add_interaction(
        self,
        user_message: BaseMessage,
        assistant_message: BaseMessage,
        workflow_instance_id: str,
    ):
        """
        Adds a user-assistant interaction to the memory storage.

        Args:
            user_message (BaseMessage): The user message.
            assistant_message (BaseMessage): The assistant message.
            workflow_instance_id: Workflow instance id for this interaction.
        """
        pass

    @abstractmethod
    def get_messages(self, workflow_instance_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all messages from the memory storage for the given workflow instance.

        Args:
            workflow_instance_id: Workflow instance id to retrieve messages for.

        Returns:
            List[Dict[str, Any]]: A list of all stored messages as dictionaries.

        Note:
            This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def reset_memory(self, workflow_instance_id: str):
        """
        Clears all messages from the memory storage for the given workflow instance.

        Args:
            workflow_instance_id: Workflow instance id to reset.

        Note:
            This method must be implemented by subclasses.
        """
        pass

    def purge_memory(self, workflow_instance_id: str) -> None:
        """
        Permanently remove all stored messages for the given workflow instance.
        Default implementation calls reset_memory; stores may override for stronger semantics.

        Args:
            workflow_instance_id: Workflow instance id to purge.
        """
        self.reset_memory(workflow_instance_id)

    @staticmethod
    def _convert_to_dict(message: Union[Dict, BaseMessage]) -> Dict:
        """
        Converts a BaseMessage to a dictionary if necessary.

        Args:
            message (Union[Dict, BaseMessage]): The message to potentially convert.

        Returns:
            Dict: The message as a dictionary.
        """
        return message.model_dump() if isinstance(message, BaseMessage) else message
