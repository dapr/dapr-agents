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

from typing import TYPE_CHECKING, Any

from .anthropic.chat import AnthropicChatClient
from .dapr import DaprChatClient
from .elevenlabs import ElevenLabsSpeechClient
from .huggingface.chat import HFHubChatClient
from .iflytek.chat import IFlytekChatClient
from .mistral.chat import MistralChatClient
from .nvidia.chat import NVIDIAChatClient
from .nvidia.embeddings import NVIDIAEmbeddingClient
from .openai.audio import OpenAIAudioClient
from .openai.chat import OpenAIChatClient
from .openai.embeddings import OpenAIEmbeddingClient

__all__ = [
    "OpenAIChatClient",
    "OpenAIAudioClient",
    "OpenAIEmbeddingClient",
    "HFHubChatClient",
    "IFlytekChatClient",
    "MistralChatClient",
    "NVIDIAChatClient",
    "NVIDIAEmbeddingClient",
    "ElevenLabsSpeechClient",
    "DaprChatClient",
    "AnthropicChatClient",
    "LiteLLMChatClient",
]

if TYPE_CHECKING:
    from .litellm.chat import LiteLLMChatClient


def __getattr__(name: str) -> Any:
    # LiteLLM is a heavy optional dependency (~190 MB RSS, ~2100 modules on
    # import). Load it lazily so merely importing dapr_agents.llm — e.g. to use
    # DaprChatClient — does not pay that cost or require litellm to be installed.
    if name == "LiteLLMChatClient":
        from .litellm.chat import LiteLLMChatClient

        return LiteLLMChatClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
