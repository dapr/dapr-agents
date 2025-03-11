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
from dapr_agents import NVIDIAChatClient
from dapr_agents.types import UserMessage
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Basic chat completion
llm = NVIDIAChatClient()
response = llm.generate("Name a famous dog!")

if len(response.get_content()) > 0:
    print("Response: ", response.get_content())

# Chat completion using a prompty file for context
llm = NVIDIAChatClient.from_prompty('basic.prompty')
response = llm.generate(input_data={"question":"What is your name?"})

if len(response.get_content()) > 0:
    print("Response with prompty: ", response.get_content())

# Chat completion with user input
llm = NVIDIAChatClient()
response = llm.generate(messages=[UserMessage("hello")])


if len(response.get_content()) > 0 and "hello" in response.get_content().lower():
    print("Response with user input: ", response.get_content())
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

from dapr_agents import NVIDIAChatClient
from dapr_agents.types import UserMessage
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Define our data model
class Dog(BaseModel):
    name: str
    breed: str
    reason: str

# Initialize the chat client
llm = NVIDIAChatClient(
    model="meta/llama-3.1-8b-instruct"
)

# Get structured response
response = llm.generate(
    messages=[UserMessage("One famous dog in history.")],
    response_format=Dog
)

print(json.dumps(response.model_dump(), indent=2))
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