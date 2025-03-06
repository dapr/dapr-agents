# LLM Call with Dapr Agents

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

### 1. Text Completion

Run the basic text completion example:

<!-- STEP
name: Run text completion example
expected_stdout_lines:
  - "Response:"
timeout_seconds: 30
output_match_mode: substring
-->
```bash
python text_completion.py
```
<!-- END_STEP -->

The script demonstrates basic usage of Dapr Agents' OpenAIChatClient for text generation:

```python
from dapr_agents import OpenAIChatClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize the chat client and call
llm = OpenAIChatClient()
response = llm.generate("Name a famous dog!")

if len(response.get_content()) > 0:
    print("Response: ", response.get_content())
```

**Expected output:** The LLM will respond with the name of a famous dog (e.g., "Lassie", "Hachiko", etc.).

### 2. Structured Output

Run the structured output example:

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

from dapr_agents import OpenAIChatClient
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
llm = OpenAIChatClient()

# Get structured response
response = llm.generate(
    messages=[UserMessage("One famous dog in history.")],
    response_format=Dog
)

print(json.dumps(response.model_dump(), indent=2))
```

**Expected output:** A structured Dog object with name, breed, and reason fields (e.g., `Dog(name='Hachiko', breed='Akita', reason='Known for his remarkable loyalty...')`)

## Key Concepts

- **OpenAIChatClient**: The interface for interacting with OpenAI's language models
- **generate()**: The primary method for getting responses from LLMs
- **response_model**: Using Pydantic models to structure LLM outputs
- **get_content()**: Extracting plain text from LLM responses

## Dapr Integration

While these examples don't explicitly use Dapr's distributed capabilities, Dapr Agents provides:

- **Unified API**: Consistent interfaces for different LLM providers
- **Type Safety**: Structured data extraction and validation
- **Integration Path**: Foundation for building more complex, distributed LLM applications

In later quickstarts, you'll see how these LLM interactions integrate with Dapr's building blocks.

## Troubleshooting

1. **Authentication Errors**: If you encounter authentication failures, check your OpenAI API key in the `.env` file
2. **Structured Output Errors**: If the model fails to produce valid structured data, try refining your model or prompt
3. **Module Not Found**: Ensure you've activated your virtual environment and installed the requirements

## Next Steps

After completing these examples, move on to the [Agent Tool Call quickstart](../03-agent-tool-call) to learn how to build agents that can use tools to interact with external systems.