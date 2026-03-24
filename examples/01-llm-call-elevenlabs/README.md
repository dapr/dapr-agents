<!--
Copyright 2026 The Dapr Authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Elevenlabs LLM calls with Dapr Agents

This example demonstrates how to use Dapr Agents to convert text to speech using the ElevenLabs API. You'll learn how to generate natural-sounding audio from text, configure voices and models, and save the resulting audio to a file in your Python application.

## Prerequisites

- Python >= 3.11
- uv package manager
- Elevenlabs API key

## Environment Setup

```bash
uv venv
# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
uv sync --active
```

## Configuration

Create a `.env` file in the project root:

```env
ELEVENLABS_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual Elevenlabs API key.

## Examples

### Audio
You can use the `ElevenLabsSpeechClient` in `dapr-agents` for text to speech capabilities of the Elevenlabs Audio API.

**1. Run the text to speech example:**
```bash
uv run python text_to_speech.py
```

## Key Concepts

- **ElevenLabsSpeechClient**: The interface for interacting with Elevenlabs' language models
- **create_speech()**: The primary method for text to speech capabilities

## Dapr Integration

While these examples don't explicitly use Dapr's distributed capabilities, Dapr Agents provides:

- **Unified API**: Consistent interfaces for different LLM providers
- **Type Safety**: Structured data extraction and validation
- **Integration Path**: Foundation for building more complex, distributed LLM applications

In later examples, you'll see how these LLM interactions integrate with Dapr's building blocks.

## Troubleshooting

1. **Authentication Errors**: If you encounter authentication failures, check your Elevenlabs API key in the `.env` file
2. **Structured Output Errors**: If the model fails to produce valid structured data, try refining your model or prompt
3. **Module Not Found**: Ensure you've activated your virtual environment and installed the requirements

## Next Steps

After completing these examples, move on to the [Durable Agent Tool Call example](../02-durable-agent-tool-call) to learn how to build agents that can use tools to interact with external systems.
