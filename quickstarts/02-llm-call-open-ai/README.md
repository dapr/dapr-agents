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
```bash
python text_completion.py
```

The script demonstrates basic usage of Dapr Agents' OpenAIChatClient for text generation.

**2. Expected output:** The LLM will respond with the name of a famous dog (e.g., "Lassie", "Hachiko", etc.).

**Run the structured text completion example:**
```bash
python structured_completion.py
```

This example shows how to use Pydantic models to get structured data from LLMs.

**Expected output:** A structured Dog object with name, breed, and reason fields (e.g., `Dog(name='Hachiko', breed='Akita', reason='Known for his remarkable loyalty...')`)

### Streaming

Our OpenAI chat client also support streaming responses, where you can process partial results as they arrive. Below are two examples:

**1. Basic Streaming Example**

Run the `text_completion_stream.py` script to see token‐by‐token output:

```bash
python text_completion_stream.py
```

This will print each partial chunk as it arrives, so you can build up the full answer in real time.

**2. Streaming with Tool Calls:**

Use `text_completion_stream_with_tools.py` to combine streaming with function‐call “tools”:

```bash
python text_completion_stream_with_tools.py
```

Here, the model can decide to call your `add_numbers` function mid‐stream, and you’ll see those calls (and their results) as they come in.

### Audio
You can use the OpenAIAudioClient in `dapr-agents` for basic tasks with the OpenAI Audio API. We will explore:

- Generating speech from text and saving it as an MP3 file.
- Transcribing audio to text.
- Translating audio content to English.

**1. Run the text to speech example:**
```bash
python text_to_speech.py
```

**2. Run the speech to text transcription example:**
```bash
python audio_transcription.py
```


**2. Run the speech to text translation example:**

[//]: # (name: Run audio translation example)

[//]: # (expected_stdout_lines:)

[//]: # (  - "Translation:")

[//]: # (  - "Success! The translation contains at least 5 out of 6 words.")

[//]: # (-->)

[//]: # (```bash)

[//]: # (python audio_translation.py)

[//]: # (```)

### Embeddings
You can use the `OpenAIEmbedder` in dapr-agents for generating text embeddings.

**1. Embeddings a single text:**
```bash
python embeddings.py
```

## Troubleshooting
1. **Authentication Errors**: If you encounter authentication failures, check your OpenAI API key in the `.env` file
2. **Structured Output Errors**: If the model fails to produce valid structured data, try refining your model or prompt
3. **Module Not Found**: Ensure you've activated your virtual environment and installed the requirements

## Next Steps

After completing these examples, move on to the [Agent Tool Call quickstart](../03-agent-tool-call/README.md) to learn how to build agents that can use tools to interact with external systems.