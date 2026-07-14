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
from typing import TYPE_CHECKING, Any

from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.configs import (
    AgentApprovalConfig,
    AgentMetadataSchema,
    AgentMetadata,
    PubSubMetadata,
    MemoryMetadata,
    ToolMetadata,
    RegistryMetadata,
    LLMMetadata,
)
from dapr_agents.agents.executors import (
    AgentEvent,
    AgentEventType,
    AgentExecutorBase,
    EchoAgentExecutor,
)
from dapr_agents.agents.schemas import ApprovalRequiredEvent, ApprovalResponseEvent
from dapr_agents.types import ActivationCallback, ActivationContext
from dapr_agents.hooks import (
    AfterHook,
    AfterLLMHook,
    AfterToolHook,
    BeforeHook,
    BeforeLLMHook,
    BeforeToolHook,
    Deny,
    HookContext,
    HookDecision,
    Hooks,
    LLMHookContext,
    Mutate,
    Proceed,
    RequireApproval,
    Skip,
    ToolHookContext,
)
from dapr_agents.executors import DockerCodeExecutor, LocalCodeExecutor
from dapr_agents.llm.anthropic import AnthropicChatClient
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
    "AgentEvent",
    "AgentEventType",
    "AgentExecutorBase",
    "EchoAgentExecutor",
    "DockerCodeExecutor",
    "LocalCodeExecutor",
    "AnthropicChatClient",
    "LiteLLMChatClient",
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
    "ActivationCallback",
    "ActivationContext",
    "call_agent",
    "trigger_agent",
    "AgentApprovalConfig",
    "ApprovalRequiredEvent",
    "ApprovalResponseEvent",
    "Hooks",
    "HookContext",
    "HookDecision",
    "LLMHookContext",
    "ToolHookContext",
    "BeforeHook",
    "AfterHook",
    "BeforeLLMHook",
    "AfterLLMHook",
    "BeforeToolHook",
    "AfterToolHook",
    "Proceed",
    "Skip",
    "Mutate",
    "RequireApproval",
    "Deny",
    "AgentMetadataSchema",
    "AgentMetadata",
    "PubSubMetadata",
    "MemoryMetadata",
    "ToolMetadata",
    "RegistryMetadata",
    "LLMMetadata",
]

if TYPE_CHECKING:
    from dapr_agents.llm.litellm import LiteLLMChatClient


def __getattr__(name: str) -> Any:
    # LiteLLM is a heavy optional dependency (~190 MB RSS, ~2100 modules on
    # import). Load it lazily so merely importing dapr_agents does not pay that
    # cost in every agent process. Eager import here previously multiplied
    # memory use across multi-agent workflows and starved co-located model
    # servers on constrained CI runners.
    if name == "LiteLLMChatClient":
        from dapr_agents.llm.litellm import LiteLLMChatClient

        return LiteLLMChatClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    __version__ = version("dapr-agents")
except PackageNotFoundError:
    # This should only happen during development
    __version__ = "0.0.0.dev0"
