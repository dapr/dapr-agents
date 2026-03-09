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

from dotenv import load_dotenv

from dapr_agents import OpenAIChatClient
from dapr_agents.types.message import LLMChatResponseChunk
from typing import Iterator
import logging

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env
load_dotenv()

# Basic chat completion
llm = OpenAIChatClient()

response: Iterator[LLMChatResponseChunk] = llm.generate(
    "Name a famous dog!", stream=True
)

for chunk in response:
    if chunk.result.content:
        print(chunk.result.content, end="", flush=True)
