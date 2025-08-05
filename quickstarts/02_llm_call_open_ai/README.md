# OpenAI LLM calls with Dapr Agents

This quickstart demonstrates how to use Dapr Agents' LLM capabilities to interact with language models and generate both free-form text and structured data. You'll learn how to make basic calls to LLMs and how to extract structured information in a type-safe manner.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key

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
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

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

The script demonstrates basic usage of Dapr Agents' OpenAIChatClient for text generation:

```python
from dotenv import load_dotenv

from dapr_agents import OpenAIChatClient
from dapr_agents.types import LLMChatResponse, UserMessage

# Load environment variables from .env
load_dotenv()

# Basic chat completion
llm = OpenAIChatClient()
response: LLMChatResponse = llm.generate("Name a famous dog!")

if response.get_message() is not None:
    print("Response: ", response.get_message().content)

# Chat completion using a prompty file for context
llm = OpenAIChatClient.from_prompty("basic.prompty")
response: LLMChatResponse = llm.generate(input_data={"question": "What is your name?"})

if response.get_message() is not None:
    print("Response with prompty: ", response.get_message().content)

# Chat completion with user input
llm = OpenAIChatClient()
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

from dapr_agents import OpenAIChatClient
from dapr_agents.types import UserMessage

# Load environment variables from .env
load_dotenv()


# Define our data model
class Dog(BaseModel):
    name: str
    breed: str
    reason: str


# Initialize the chat client
llm = OpenAIChatClient()

# Get structured response
response: Dog = llm.generate(
    messages=[UserMessage("One famous dog in history.")], response_format=Dog
)

print(json.dumps(response.model_dump(), indent=2))
```

**Expected output:** A structured Dog object with name, breed, and reason fields (e.g., `Dog(name='Hachiko', breed='Akita', reason='Known for his remarkable loyalty...')`)

### Streaming

Our OpenAI chat client also support streaming responses, where you can process partial results as they arrive. Below are two examples:

**1. Basic Streaming Example**

Run the `text_completion_stream.py` script to see token‐by‐token output:

```bash
python text_completion_stream.py
```

The scripts:

```python
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

### Audio
You can use the OpenAIAudioClient in `dapr-agents` for basic tasks with the OpenAI Audio API. We will explore:

- Generating speech from text and saving it as an MP3 file.
- Transcribing audio to text.
- Translating audio content to English.

**1. Run the text to speech example:**


<!-- STEP
name: Run audio generation example
expected_stdout_lines:
  - "Audio saved to speech.mp3"
  - "File speech.mp3 has been deleted."
-->
```bash
python text_to_speech.py
```
<!-- END_STEP -->

**2. Run the speech to text transcription example:**

<!-- STEP
name: Run audio transcription example
expected_stdout_lines:
  - "Transcription:"
  - "Success! The transcription contains at least 5 out of 7 words."
output_match_mode: substring
-->
```bash
python audio_transcription.py
```
<!-- END_STEP -->


**2. Run the speech to text translation example:**

[//]: # (<!-- STEP)

[//]: # (name: Run audio translation example)

[//]: # (expected_stdout_lines:)

[//]: # (  - "Translation:")

[//]: # (  - "Success! The translation contains at least 5 out of 6 words.")

[//]: # (-->)

[//]: # (```bash)

[//]: # (python audio_translation.py)

[//]: # (```)

[//]: # (<!-- END_STEP -->)

### Embeddings
You can use the `OpenAIEmbedder` in dapr-agents for generating text embeddings.

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

## Troubleshooting
1. **Authentication Errors**: If you encounter authentication failures, check your OpenAI API key in the `.env` file
2. **Structured Output Errors**: If the model fails to produce valid structured data, try refining your model or prompt
3. **Module Not Found**: Ensure you've activated your virtual environment and installed the requirements

## Next Steps

After completing these examples, move on to the [Agent Tool Call quickstart](../03-agent-tool-call/README.md) to learn how to build agents that can use tools to interact with external systems.