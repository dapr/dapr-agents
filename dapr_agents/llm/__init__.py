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

from .base import LLMClientBase
from .chat import ChatClientBase
from .dapr import DaprChatClient
from .elevenlabs import ElevenLabsSpeechClient
from .huggingface.chat import HFHubChatClient
from .huggingface.client import HFHubInferenceClientBase
from .nvidia.chat import NVIDIAChatClient
from .nvidia.client import NVIDIAClientBase
from .nvidia.embeddings import NVIDIAEmbeddingClient
from .openai.audio import OpenAIAudioClient
from .openai.chat import OpenAIChatClient
from .openai.client import AzureOpenAIClient, OpenAIClient
from .openai.embeddings import OpenAIEmbeddingClient

__all__ = [
    "LLMClientBase",
    "ChatClientBase",
    "OpenAIClient",
    "AzureOpenAIClient",
    "OpenAIChatClient",
    "OpenAIAudioClient",
    "OpenAIEmbeddingClient",
    "HFHubInferenceClientBase",
    "HFHubChatClient",
    "NVIDIAClientBase",
    "NVIDIAChatClient",
    "NVIDIAEmbeddingClient",
    "ElevenLabsSpeechClient",
    "DaprChatClient",
]
