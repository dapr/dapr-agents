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
