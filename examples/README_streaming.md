# Dapr Streaming Chat Completion

This guide demonstrates how to use the new streaming capability with Dapr LLM components in the `dapr-agents` library.

## Overview

The Dapr streaming functionality allows you to receive real-time responses from LLM components, providing a better user experience for chat applications and reducing perceived latency. This implementation leverages the new `converse_stream_json` method from the Dapr Python SDK, which provides JSON-formatted responses that are compatible with common LLM APIs like OpenAI.

## Features

- ✅ **Real-time streaming**: Get responses as they're generated
- ✅ **Context continuity**: Support for context IDs to maintain conversation state
- ✅ **Usage tracking**: Monitor token consumption in real-time
- ✅ **PII scrubbing**: Optional scrubbing of sensitive information
- ✅ **Temperature control**: Fine-tune response creativity
- ✅ **Error handling**: Robust error handling for production use

## Usage

### Basic Streaming Example

```python
from dapr_agents.llm.dapr import DaprChatClient
import os

# Set up your LLM component
os.environ["DAPR_LLM_COMPONENT_DEFAULT"] = "my-llm-component"

# Initialize client
client = DaprChatClient()

# Prepare messages
messages = [
    {"role": "user", "content": "Tell me a story about AI"}
]

# Stream the response
response_stream = client.generate(
    messages=messages,
    stream=True,
    temperature=0.7
)

# Process streaming chunks
print("Assistant: ", end="", flush=True)
for chunk in response_stream:
    if chunk.get("type") == "content":
        print(chunk["data"], end="", flush=True)
    elif chunk.get("type") == "final_usage":
        usage = chunk["data"]
        print(f"\nTokens used: {usage.get('total_tokens')}")
```

### Advanced Streaming with Context

```python
# Continuing a conversation with context
context_id = "my-conversation-123"

response_stream = client.generate(
    messages=messages,
    stream=True,
    context_id=context_id,
    temperature=0.7,
    scrubPII=True  # Enable PII scrubbing
)

for chunk in response_stream:
    if chunk.get("type") == "content":
        print(chunk["data"], end="", flush=True)
    elif chunk.get("type") == "context_id":
        context_id = chunk["data"]  # Save for next request
```

### Comparison: Streaming vs Non-Streaming

```python
import time

# Non-streaming (traditional)
start_time = time.time()
response = client.generate(messages=messages, stream=False)
print(f"Non-streaming took: {time.time() - start_time:.2f} seconds")
print(response.choices[0].message.content)

# Streaming (real-time)
start_time = time.time()
first_chunk_time = None
response_stream = client.generate(messages=messages, stream=True)

for chunk in response_stream:
    if chunk.get("type") == "content" and first_chunk_time is None:
        first_chunk_time = time.time()
        print(f"First chunk received in: {first_chunk_time - start_time:.2f} seconds")
    # Process chunk...
```

## Configuration

### Environment Variables

- `DAPR_LLM_COMPONENT_DEFAULT`: Default LLM component name to use

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stream` | `bool` | `False` | Enable streaming responses |
| `context_id` | `str` | `None` | Context ID for conversation continuity |
| `temperature` | `float` | `None` | Controls randomness (0.0-1.0) |
| `scrubPII` | `bool` | `False` | Enable PII scrubbing |
| `llm_component` | `str` | `None` | Override default LLM component |

## Error Handling

```python
try:
    response_stream = client.generate(
        messages=messages,
        stream=True,
        temperature=0.7
    )
    
    for chunk in response_stream:
        # Process chunk
        pass
        
except Exception as e:
    print(f"Streaming error: {e}")
    # Fallback to non-streaming
    response = client.generate(messages=messages, stream=False)
```

## Prerequisites

1. **Dapr Sidecar**: Ensure Dapr sidecar is running
2. **LLM Component**: Configure your LLM component in Dapr
3. **Environment**: Set `DAPR_LLM_COMPONENT_DEFAULT` environment variable

### Starting Dapr Sidecar with LLM Component

```bash
# Example with OpenAI component
dapr run --app-id my-app --components-path ./components --app-port 3000 python my_app.py
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure Dapr sidecar is running
2. **Component Not Found**: Check LLM component configuration
3. **Environment Variable Missing**: Set `DAPR_LLM_COMPONENT_DEFAULT`

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your streaming code here
```

## Performance Considerations

- **Streaming**: Better for real-time applications, lower perceived latency
- **Non-streaming**: Better for batch processing, simpler error handling
- **Context IDs**: Use for multi-turn conversations to maintain state
- **Token Usage**: Monitor streaming usage to avoid unexpected costs

## Examples

### Basic Echo Example
- [`dapr_streaming_example.py`](./dapr_streaming_example.py): Basic streaming with echo component

### OpenAI Streaming Examples

For real-world usage with OpenAI models:

- [`test_openai_streaming.py`](../test_openai_streaming.py): Comprehensive OpenAI streaming with direct Dapr client
- [`test_openai_agents_streaming.py`](../test_openai_agents_streaming.py): OpenAI streaming through dapr-agents API

#### OpenAI Component Setup

First, create an OpenAI component configuration:

```yaml
# components/openai-conversation.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: openai
spec:
  type: conversation.openai
  metadata:
    - name: key
      value: "${OPENAI_API_KEY}"
    - name: model
      value: "gpt-4-turbo"
    - name: cacheTTL
      value: "10m"
    - name: temperature
      value: "0.7"
    - name: maxTokens
      value: "1000"
```

#### Quick OpenAI Streaming Example

```python
import os
from dapr_agents.llm.dapr import DaprChatClient

# Ensure OpenAI API key is set
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize client with OpenAI component
client = DaprChatClient(llm_component="openai")

messages = [{"role": "user", "content": "Write a haiku about programming"}]

print("Assistant: ", end="", flush=True)
response_stream = client.generate(messages=messages, stream=True)

for chunk in response_stream:
    if chunk.get("type") == "content":
        print(chunk["data"], end="", flush=True)
    elif chunk.get("type") == "final_usage":
        usage = chunk["data"]
        print(f"\nTokens: {usage.get('total_tokens')}")
```

#### Running OpenAI Examples

1. **Set up environment**:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

2. **Start local development Dapr**:
   ```bash
   ./start_dapr.sh --dev
   ```

3. **Run the examples**:
   ```bash
   # High-level dapr-agents API
   python test_openai_agents_streaming.py
   
   # Lower-level Dapr client API
   python test_openai_streaming.py
   ```

- Check the `quickstarts/` directory for more advanced use cases

## Contributing

To contribute improvements to the streaming functionality:

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Submit a pull request

## Support

For issues with streaming functionality:

1. Check the troubleshooting section above
2. Enable debug logging
3. Check Dapr sidecar logs
4. Open an issue with reproduction steps 