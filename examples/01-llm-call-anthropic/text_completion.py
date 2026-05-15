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

from dapr_agents import AnthropicChatClient
from dapr_agents.types import LLMChatResponse, UserMessage

# Load environment variables from .env (expects ANTHROPIC_API_KEY)
load_dotenv()

# Basic chat completion
llm = AnthropicChatClient()
response: LLMChatResponse = llm.generate("Name a famous dog!")

if response.get_message() is not None:
    print("Response: ", response.get_message().content)

# Chat completion with explicit user input
llm = AnthropicChatClient()
response: LLMChatResponse = llm.generate(messages=[UserMessage("hello")])

if (
    response.get_message() is not None
    and "hello" in (response.get_message().content or "").lower()
):
    print("Response with user input: ", response.get_message().content)

# Native Anthropic features: extended thinking
llm = AnthropicChatClient(model="claude-opus-4-5")
response = llm.generate(
    "What is 17 * 23? Think step by step.",
    thinking={"type": "enabled", "budget_tokens": 1024},
    max_tokens=2048,
)
if response.get_message() is not None:
    print("Response with thinking: ", response.get_message().content)
