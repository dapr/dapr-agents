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

    messages: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Messages stored per workflow instance id as dictionaries.",
    )

    def add_message(
        self, message: Union[Dict[str, Any], BaseMessage], workflow_instance_id: str
    ) -> None:
        """
        Adds a single message to the end of the memory list for the given id.

        Args:
            message: The message to add to the memory.
            workflow_instance_id: Workflow instance id for this message.
        """
        self.messages.setdefault(workflow_instance_id, []).append(self._convert_to_dict(message))

    def add_messages(
        self,
        messages: List[Union[Dict[str, Any], BaseMessage]],
        workflow_instance_id: str,
    ) -> None:
        """
        Adds multiple messages to the memory for the given id.

        Args:
            messages: A list of messages to add to the memory.
            workflow_instance_id: Workflow instance id for these messages.
        """
        self.messages.setdefault(workflow_instance_id, []).extend(
            self._convert_to_dict(msg) for msg in messages
        )

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
        Retrieves a copy of the messages stored for the given id.

        Args:
            workflow_instance_id: Workflow instance id to retrieve messages for.

        Returns:
            A list containing copies of the stored messages as dictionaries
            (empty when the id has no messages).
        """
        return self.messages.get(workflow_instance_id, []).copy()

    def reset_memory(self, workflow_instance_id: str) -> None:
        """
        Clears the messages stored for the given id only.

        Args:
            workflow_instance_id: Workflow instance id to reset.
        """
        self.messages.pop(workflow_instance_id, None)
