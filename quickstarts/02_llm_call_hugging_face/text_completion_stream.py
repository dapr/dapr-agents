from dotenv import load_dotenv

from dapr_agents import HFHubChatClient
from dapr_agents.types.message import LLMChatResponseChunk
from typing import Iterator
import logging

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env
load_dotenv()

# Basic chat completion
llm = HFHubChatClient(model="HuggingFaceTB/SmolLM3-3B")

response: Iterator[LLMChatResponseChunk] = llm.generate(
    "Name a famous dog!", stream=True
)

for chunk in response:
    if chunk.result.content:
        print(chunk.result.content, end="", flush=True)
