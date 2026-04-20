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

from importlib.metadata import version, PackageNotFoundError
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.configs import (
    AgentMetadataSchema,
    AgentMetadata,
    PubSubMetadata,
    MemoryMetadata,
    ToolMetadata,
    RegistryMetadata,
    LLMMetadata,
)
from dapr_agents.executors import DockerCodeExecutor, LocalCodeExecutor
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.llm.elevenlabs import ElevenLabsSpeechClient
from dapr_agents.llm.huggingface import HFHubChatClient
from dapr_agents.llm.nvidia import NVIDIAChatClient, NVIDIAEmbeddingClient
from dapr_agents.llm.openai import (
    OpenAIAudioClient,
    OpenAIChatClient,
    OpenAIEmbeddingClient,
)
from dapr_agents.tool import AgentTool, tool
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.workflow.utils.core import call_agent, trigger_agent

__all__ = [
    "DurableAgent",
    "DockerCodeExecutor",
    "LocalCodeExecutor",
    "ElevenLabsSpeechClient",
    "DaprChatClient",
    "HFHubChatClient",
    "NVIDIAChatClient",
    "NVIDIAEmbeddingClient",
    "OpenAIAudioClient",
    "OpenAIChatClient",
    "OpenAIEmbeddingClient",
    "AgentTool",
    "tool",
    "AgentRunner",
    "call_agent",
    "trigger_agent",
    "AgentMetadataSchema",
    "AgentMetadata",
    "PubSubMetadata",
    "MemoryMetadata",
    "ToolMetadata",
    "RegistryMetadata",
    "LLMMetadata",
]

try:
    __version__ = version("dapr-agents")
except PackageNotFoundError:
    # This should only happen during development
    __version__ = "0.0.0.dev0"
