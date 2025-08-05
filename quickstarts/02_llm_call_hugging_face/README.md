# LLM calls with Hugging Face

This quickstart demonstrates how to use Dapr Agents' LLM capabilities to interact with the Hugging Face Hub language models and generate both free-form text and structured data. You'll learn how to make basic calls to LLMs and how to extract structured information in a type-safe manner.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager

## Environment Setup

```bash
# Create a virtual environment
python3.10 -m venv .venv

# Activate the virtual environment 
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Examples

### Text

**1. Run the basic text completion example:**

<!-- STEP
name: Run text completion example
expected_stdout_lines:
  - "Response:"
  - "Response with prompty:"
  - "Response with user input:"
timeout_seconds: 30
output_match_mode: substring
-->
```bash
python text_completion.py
```
<!-- END_STEP -->

The script demonstrates basic usage of the DaprChatClient for text generation:

```python
from dotenv import load_dotenv

from dapr_agents.llm import HFHubChatClient
from dapr_agents.types import LLMChatResponse, UserMessage

load_dotenv()

# Basic chat completion
llm = HFHubChatClient(model="HuggingFaceTB/SmolLM3-3B")
response: LLMChatResponse = llm.generate("Name a famous dog!")

if response.get_message() is not None:
    print("Response: ", response.get_message().content)

# Chat completion using a prompty file for context
llm = HFHubChatClient.from_prompty("basic.prompty")
response: LLMChatResponse = llm.generate(input_data={"question": "What is your name?"})

if response.get_message() is not None:
    print("Response with prompty: ", response.get_message().content)

# Chat completion with user input
llm = HFHubChatClient(model="HuggingFaceTB/SmolLM3-3B")
response: LLMChatResponse = llm.generate(messages=[UserMessage("hello")])

if response.get_message() is not None and "hello" in response.get_message().content.lower():
    print("Response with user input: ", response.get_message().content)
```

**2. Expected output:** The LLM will respond with the name of a famous dog (e.g., "Lassie", "Hachiko", etc.).

**Run the structured text completion example:**

<!-- STEP
name: Run text completion example
expected_stdout_lines:
  - '"name":'
  - '"breed":'
  - '"reason":'
timeout_seconds: 30
output_match_mode: substring
-->
```bash
python structured_completion.py
```
<!-- END_STEP -->

This example shows how to use Pydantic models to get structured data from LLMs:

```python
import json

from dotenv import load_dotenv
from pydantic import BaseModel

from dapr_agents import HFHubChatClient
from dapr_agents.types import UserMessage

# Load environment variables from .env
load_dotenv()


# Define our data model
class Dog(BaseModel):
    name: str
    breed: str
    reason: str


# Initialize the chat client
llm = HFHubChatClient(model="HuggingFaceTB/SmolLM3-3B")

# Get structured response
response: Dog = llm.generate(
    messages=[UserMessage("One famous dog in history.")], response_format=Dog
)

print(json.dumps(response.model_dump(), indent=2))
```

**Expected output:** A JSON object with name, breed, and reason fields

```
{
  "name": "Dog",
  "breed": "Siberian Husky",
  "reason": "Known for its endurance, intelligence, and loyalty, Siberian Huskies have played crucial roles in dog sledding and have been beloved companions for many."
}
```

### Streaming

Our Hugging Face chat client also support streaming responses, where you can process partial results as they arrive. Below are two examples:

**1. Basic Streaming Example**

Run the `text_completion_stream.py` script to see token‐by‐token output:

```bash
python text_completion_stream.py
```

The scripts:

```python
from dotenv import load_dotenv
from dapr_agents import HFHubChatClient
from dapr_agents.types.message import LLMChatResponseChunk
from typing import Iterator
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()

llm = HFHubChatClient(model="HuggingFaceTB/SmolLM3-3B")
response: Iterator[LLMChatResponseChunk] = llm.generate("Name a famous dog!", stream=True)

for chunk in response:
    if chunk.result.content:
        print(chunk.result.content, end="", flush=True)
```

This will print each partial chunk as it arrives, so you can build up the full answer in real time.

**2. Streaming with Tool Calls:**

Use `text_completion_stream_with_tools.py` to combine streaming with function‐call “tools”:

```bash
python text_completion_stream_with_tools.py
```

```python
from dotenv import load_dotenv
from dapr_agents import HFHubChatClient
from dapr_agents.types.message import LLMChatResponseChunk
from typing import Iterator
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()

# Initialize client
llm = HFHubChatClient(model="HuggingFaceTB/SmolLM3-3B", hf_provider="auto")

# Define a simple addition tool
def add_numbers(a: int, b: int) -> int:
    return a + b

add_tool = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two numbers together.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The first number."},
                "b": {"type": "integer", "description": "The second number."}
            },
            "required": ["a", "b"]
        }
    }
}

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Add 5 and 7 and 2 and 2."}
]

response: Iterator[LLMChatResponseChunk] = llm.generate(
    messages=messages,
    tools=[add_tool],
    stream=True
)

for chunk in response:
    print(chunk.result)
```

Here, the model can decide to call your add_numbers function mid‐stream, and you’ll see those calls (and their results) as they come in.