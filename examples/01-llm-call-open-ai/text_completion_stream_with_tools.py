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


# Define a tool for addition
def add_numbers(a: int, b: int) -> int:
    return a + b


# Define the tool function call schema
add_tool = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two numbers together.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The first number."},
                "b": {"type": "integer", "description": "The second number."},
            },
            "required": ["a", "b"],
        },
    },
}

# Define messages for the chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Add 5 and 7 and 2 and 2."},
]

response: Iterator[LLMChatResponseChunk] = llm.generate(
    messages=messages, tools=[add_tool], stream=True
)

for chunk in response:
    print(chunk.result)
