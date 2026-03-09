#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

    def add_message(
        self, message: Union[Dict[str, Any], BaseMessage], workflow_instance_id: str
    ) -> None:
        """
        Adds a single message to the end of the memory list.

        Args:
            message: The message to add to the memory.
            workflow_instance_id: Workflow instance id for this message (ignored for single-list storage).
        """
        self.messages.append(self._convert_to_dict(message))

    def add_messages(
        self,
        messages: List[Union[Dict[str, Any], BaseMessage]],
        workflow_instance_id: str,
    ) -> None:
        """
        Adds multiple messages to the memory by appending each message to the list.

        Args:
            messages: A list of messages to add to the memory.
            workflow_instance_id: Workflow instance id for these messages (ignored for single-list storage).
        """
        self.messages.extend(self._convert_to_dict(msg) for msg in messages)

    def add_interaction(
        self,
        user_message: BaseMessage,
        assistant_message: BaseMessage,
        workflow_instance_id: str,
    ) -> None:
        """
        Adds a user-assistant interaction to the memory storage.

        Args:
            user_message: The user message.
            assistant_message: The assistant message.
            workflow_instance_id: Workflow instance id for this interaction.
        """
        self.add_messages([user_message, assistant_message], workflow_instance_id)

    def get_messages(self, workflow_instance_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves a copy of all messages stored in the memory.

        Args:
            workflow_instance_id: Workflow instance id to retrieve messages for (ignored for single-list storage).

        Returns:
            A list containing copies of all stored messages as dictionaries.
        """
        return self.messages.copy()

    def reset_memory(self, workflow_instance_id: str) -> None:
        """
        Clears all messages stored in the memory.

        Args:
            workflow_instance_id: Workflow instance id to reset (ignored for single-list storage).
        """
        self.messages.clear()
