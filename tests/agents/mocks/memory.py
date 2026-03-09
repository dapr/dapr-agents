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

from dapr_agents.memory import MemoryBase
from dapr_agents.types import BaseMessage, MessageContent
from typing import List
from unittest.mock import Mock
from pydantic import PrivateAttr


class DummyVectorMemory(MemoryBase):
    """Mock vector memory for testing."""

    _vector_store = PrivateAttr()

    def __init__(self, vector_store):
        super().__init__()
        self._vector_store = vector_store

    def get_messages(self, workflow_instance_id: str, query_embeddings=None):
        return [Mock(spec=MessageContent)]

    def add_message(self, message: BaseMessage, workflow_instance_id: str):
        pass

    def add_messages(self, messages: List[BaseMessage], workflow_instance_id: str):
        pass

    def add_interaction(
        self,
        user_message: BaseMessage,
        assistant_message: BaseMessage,
        workflow_instance_id: str,
    ):
        pass

    def reset_memory(self, workflow_instance_id: str):
        pass
