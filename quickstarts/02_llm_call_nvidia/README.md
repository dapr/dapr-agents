# Nvidia LLM calls with Dapr Agents

This quickstart demonstrates how to use Dapr Agents' LLM capabilities to interact with language models and generate both free-form text and structured data. You'll learn how to make basic calls to LLMs and how to extract structured information in a type-safe manner.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- Nvidia API key

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

## Configuration

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual Nvidia key.

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

The script demonstrates basic usage of Dapr Agents' NVIDIAChatClient for text generation:

```python
from dotenv import load_dotenv

from dapr_agents import NVIDIAChatClient
from dapr_agents.types import LLMChatResponse, UserMessage

# Load environment variables from .env
load_dotenv()

# Basic chat completion
llm = NVIDIAChatClient()
response: LLMChatResponse = llm.generate("Name a famous dog!")

if response.get_message() is not None:
    print("Response: ", response.get_message().content)

# Chat completion using a prompty file for context
llm = NVIDIAChatClient.from_prompty("basic.prompty")
response: LLMChatResponse = llm.generate(input_data={"question": "What is your name?"})

if response.get_message() is not None:
    print("Response with prompty: ", response.get_message().content)

# Chat completion with user input
llm = NVIDIAChatClient()
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

from dapr_agents import NVIDIAChatClient
from dapr_agents.types import UserMessage

# Load environment variables from .env
load_dotenv()


# Define our data model
class Dog(BaseModel):
    name: str
    breed: str
    reason: str


# Initialize the chat client
llm = NVIDIAChatClient(model="meta/llama-3.1-8b-instruct")

# Get structured response
response: Dog = llm.generate(
    messages=[UserMessage("One famous dog in history.")], response_format=Dog
)
```

**Expected output:** A structured Dog object with name, breed, and reason fields (e.g., `Dog(name='Hachiko', breed='Akita', reason='Known for his remarkable loyalty...')`)

### Embeddings
You can use the `NVIDIAEmbedder` in dapr-agents for generating text embeddings.

**1. Embeddings a single text:**
<!-- STEP
name: Run audio transcription example
expected_stdout_lines:
  - "Embedding (first 5 values):"
  - "Text 1 embedding (first 5 values):"
  - "Text 2 embedding (first 5 values):"
output_match_mode: substring
-->
```bash
python embeddings.py
```
<!-- END_STEP -->

### Streaming

Our NVIDIA chat client also support streaming responses, where you can process partial results as they arrive. Below are two examples:

**1. Basic Streaming Example**

Run the `text_completion_stream.py` script to see token‐by‐token output:

```bash
python text_completion_stream.py
```

The scripts:

```python
from dotenv import load_dotenv

from dapr_agents import NVIDIAChatClient
from dapr_agents.types.message import LLMChatResponseChunk
from typing import Iterator
import logging

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env
load_dotenv()

# Basic chat completion
llm = NVIDIAChatClient()

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

from dapr_agents import NVIDIAChatClient
from dapr_agents.types.message import LLMChatResponseChunk
from typing import Iterator
import logging

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env
load_dotenv()

# Basic chat completion
llm = NVIDIAChatClient()

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
                "b": {"type": "integer", "description": "The second number."}
            },
            "required": ["a", "b"]
        }
    }
}

# Define messages for the chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Add 5 and 7 and 2 and 2."}
]

response: Iterator[LLMChatResponseChunk] = llm.generate(messages=messages, tools=[add_tool], stream=True)

for chunk in response:
    print(chunk.result)
```

Here, the model can decide to call your `add_numbers` function mid‐stream, and you’ll see those calls (and their results) as they come in.