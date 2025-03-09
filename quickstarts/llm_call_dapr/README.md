# OpenAI LLM calls with Dapr Agents

This quickstart demonstrates how to use Dapr Agents' LLM capabilities to interact with language models and generate both free-form text and structured data. You'll learn how to make basic calls to LLMs and how to extract structured information in a type-safe manner.

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
dapr run --app-id daprllm --resources-path components/ -- python text_completion.py
```
<!-- END_STEP -->

The script demonstrates basic usage of the DaprChatClient for text generation:

```python
import os

from dapr_agents.llm import DaprChatClient
from dapr_agents.types import UserMessage

os.environ['DAPR_LLM_COMPONENT_DEFAULT'] = 'echo'

# Basic chat completion
llm = DaprChatClient()
response = llm.generate("Name a famous dog!")

if len(response.get_content()) > 0:
    print("Response: ", response.get_content())

# Chat completion using a prompty file for context
llm = DaprChatClient.from_prompty('basic.prompty')
response = llm.generate(input_data={"question":"What is your name?"})

if len(response.get_content()) > 0:
    print("Response with prompty: ", response.get_content())

# Chat completion with user input
llm = DaprChatClient()
response = llm.generate(messages=[UserMessage("hello")])


if len(response.get_content()) > 0 and "hello" in response.get_content().lower():
    print("Response with user input: ", response.get_content())
```
