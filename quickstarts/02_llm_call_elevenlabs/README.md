# Elevenlabs LLM calls with Dapr Agents

This quickstart demonstrates how to use Dapr Agents' LLM capabilities to interact with language models and generate both free-form text and structured data. You'll learn how to make basic calls to LLMs and how to extract structured information in a type-safe manner.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- Elevenlabs API key

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
ELEVENLABS_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual Elevenlabs API key.

## Examples

### Audio
You can use the `ElevenLabsSpeechClient` in `dapr-agents` for text to speech capabilities of the Elevenlabs Audio API.

**1. Run the text to speech example:**


<!-- STEP
name: Run audio generation example
expected_stdout_lines:
  - "Audio saved to output_speech.mp3"
  - "File output_speech.mp3 has been deleted."
-->
```bash
python text_to_speech.py
```
<!-- END_STEP -->

## Key Concepts

- **ElevenLabsSpeechClient**: The interface for interacting with Elevenlabs' language models
- **create_speech()**: The primary method for text to speech capabilities

## Dapr Integration

While these examples don't explicitly use Dapr's distributed capabilities, Dapr Agents provides:

- **Unified API**: Consistent interfaces for different LLM providers
- **Type Safety**: Structured data extraction and validation
- **Integration Path**: Foundation for building more complex, distributed LLM applications

In later quickstarts, you'll see how these LLM interactions integrate with Dapr's building blocks.

## Troubleshooting

1. **Authentication Errors**: If you encounter authentication failures, check your Elevenlabs API key in the `.env` file
2. **Structured Output Errors**: If the model fails to produce valid structured data, try refining your model or prompt
3. **Module Not Found**: Ensure you've activated your virtual environment and installed the requirements

## Next Steps

After completing these examples, move on to the [Agent Tool Call quickstart](../03-agent-tool-call) to learn how to build agents that can use tools to interact with external systems.