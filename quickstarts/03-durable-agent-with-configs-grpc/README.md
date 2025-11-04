# Durable Agent with WorkflowGrpcOptions (Large Payload Support)

This quickstart demonstrates the **WorkflowGrpcOptions** configuration class for handling large payloads in Dapr Agents workflows. It shows how to increase gRPC message size limits from the default ~4 MB to 32 MB, enabling workflows to process larger tool outputs, documents, or complex data structures.

> **Note:** This example builds on the concepts introduced in [`03-durable-agent-with-configs`](../03-durable-agent-with-configs/). If you're new to Agent Configuration Classes, start there first to understand the complete configuration pattern.

## What You'll Learn

- How to configure **WorkflowGrpcOptions** for large data workflows
- When and why to increase gRPC message size limits
- How to avoid `RESOURCE_EXHAUSTED` errors with large payloads
- Practical example with all configuration classes working together

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- Dapr CLI and Docker
- OpenAI API key (or compatible LLM provider)

## Why Configure gRPC Limits?

### Default Behavior
Dapr workflow gRPC channels default to ~4 MB message limits (both send and receive). This is sufficient for most agent workflows.

### When You Need Larger Limits

Increase gRPC limits when your workflows handle:
- Large documents or files
- Image/video data
- Extensive tool outputs (e.g., large database query results)
- Large JSON payloads or API responses
- Embedded/base64-encoded content

### Error Without Proper Configuration
```
grpc._channel._InactiveRpcError: StatusCode.RESOURCE_EXHAUSTED
Details: "Received message larger than max (4194304 vs. 4096000)"
```

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

### Option 1: Using Environment Variables (Recommended)

1. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
```

2. When running with Dapr, use the helper script to resolve environment variables:
```bash
# Get environment variables from .env file
export $(grep -v '^#' ../../.env | xargs)

# Create temporary resources folder with resolved variables
temp_resources_folder=$(../resolve_env_templates.py ./components)

# Run with Dapr
dapr run \
  --app-id weather-agent \
  --resources-path $temp_resources_folder \
  -- python app.py

# Clean up when done
rm -rf $temp_resources_folder
```

### Option 2: Direct Component Configuration

Update the `key` in [components/openai.yaml](components/openai.yaml):
```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: openai
spec:
  type: conversation.openai
  metadata:
    - name: key
      value: "YOUR_OPENAI_API_KEY"
```

### Required Components

Make sure Dapr is initialized:

```bash
dapr init
```

This quickstart uses these Dapr components (in `components/` directory):
- `openai.yaml`: LLM conversation component
- `workflowstate.yaml`: Workflow state storage
- `memorystore.yaml`: Conversation memory storage
- `agentregistrystore.yaml`: Agent registry storage
- `messagepubsub.yaml`: Pub/sub messaging

## The WorkflowGrpcOptions Configuration

This quickstart focuses on the **WorkflowGrpcOptions** configuration class, which is the 7th configuration class explained in [`03-durable-agent-with-configs`](../03-durable-agent-with-configs/README.md#7-workflowgrpcoptions---large-payload-support).

### Configuration in This Example

```python
from dapr_agents.agents.configs import WorkflowGrpcOptions

# Increase limits to 32 MB (from default ~4 MB)
grpc_options = WorkflowGrpcOptions(
    max_send_message_length=32 * 1024 * 1024,     # 32 MB
    max_receive_message_length=32 * 1024 * 1024,  # 32 MB
)

# Log the configuration
logger.info(
    "Configuring workflow gRPC channel with %d MB send / %d MB receive limits",
    grpc_options.max_send_message_length // (1024 * 1024),
    grpc_options.max_receive_message_length // (1024 * 1024),
)

# Pass to DurableAgent
agent = DurableAgent(
    profile=profile,
    pubsub=pubsub,
    state=state,
    registry=registry,
    memory=memory,
    llm=llm,
    tools=tools,
    workflow_grpc=grpc_options,  # Configure gRPC limits here
)
```

### Parameters

- **`max_send_message_length`**: Maximum bytes for messages TO the workflow runtime
  - Default: ~4 MB (4,194,304 bytes)
  - This example: 32 MB (33,554,432 bytes)
  - Use case: Large tool inputs, workflow arguments

- **`max_receive_message_length`**: Maximum bytes for messages FROM the workflow runtime
  - Default: ~4 MB (4,194,304 bytes)
  - This example: 32 MB (33,554,432 bytes)
  - Use case: Large tool outputs, workflow results

## Running the Example

### Terminal 1: Start the Agent Service

```bash
dapr run \
  --app-id weather-agent \
  --resources-path ./components \
  -- python app.py
```

**Expected log output:**
```
INFO Configuring workflow gRPC channel with 32 MB send / 32 MB receive limits
INFO Starting workflow runtime with enhanced gRPC options
INFO Subscribed to topic: weather.requests
```

### Terminal 2: Publish a Message

```bash
dapr run \
  --app-id weather-client \
  --resources-path ./components \
  -- python message_client.py
```

The agent will process the weather request using the configured gRPC limits, allowing it to handle larger payloads without errors.

## What's Happening

1. **gRPC Configuration**: Before starting the agent, `WorkflowGrpcOptions` sets 32 MB limits
2. **Agent Start**: The DurableAgent initializes with enhanced gRPC channel settings
3. **Message Trigger**: Client publishes to `weather.requests` topic
4. **Workflow Execution**: Agent processes request with support for large data
5. **Tool Execution**: Weather tools can return large outputs without hitting size limits

## Code Structure

### `app.py` - Main Application

The key difference from the base config example is the gRPC configuration:

```python
# --- gRPC overrides (lift default ~4MB limit to 32MB) -----------------------
grpc_options = WorkflowGrpcOptions(
    max_send_message_length=32 * 1024 * 1024,
    max_receive_message_length=32 * 1024 * 1024,
)
logger.info(
    "Configuring workflow gRPC channel with %d MB send / %d MB receive limits",
    grpc_options.max_send_message_length // (1024 * 1024),
    grpc_options.max_receive_message_length // (1024 * 1024),
)

# Pass grpc_options to agent
agent = DurableAgent(
    profile=profile,
    pubsub=pubsub,
    state=state,
    registry=registry,
    memory=memory,
    llm=llm,
    tools=tools,
    workflow_grpc=grpc_options,  # This is the focus of this quickstart
)
```

### `weather_tools.py` - Tool Definitions

Contains the same tools as the base example:
- `get_weather(location)`: Returns weather information
- `jump(distance)`: Simulates a physical action

### `message_client.py` - Event Publisher

Publishes messages to the `weather.requests` topic to trigger workflows.

## Choosing the Right Limits

### Recommended Sizes

| Use Case | Send/Receive Limit | Notes |
|----------|-------------------|-------|
| Standard agents | 4 MB (default) | Most workflows |
| Document processing | 16 MB | Medium documents |
| Large data workflows | 32 MB | **This example** |
| Image/video processing | 64 MB+ | Very large media |

### Performance Considerations

- **Memory**: Larger limits = more memory per workflow
- **Network**: Bigger messages = longer transmission time
- **Best Practice**: Only increase when needed

### Estimating Required Size

```python
import sys

# Estimate your data size
example_output = {"large": "data" * 1000}
size_bytes = sys.getsizeof(str(example_output))
size_mb = size_bytes / (1024 * 1024)

print(f"Estimated: {size_mb:.2f} MB")
print(f"Recommended: {size_mb * 2:.2f} MB (with buffer)")
```

## Troubleshooting

**Issue:** Still getting `RESOURCE_EXHAUSTED` errors

**Solutions:**
- Verify gRPC options are set BEFORE `agent.start()`
- Check both send AND receive limits are configured
- Calculate actual message size and add 2-3x buffer

## Learn More

- **All Config Classes**: [`03-durable-agent-with-configs`](../03-durable-agent-with-configs/) - Complete guide to all 7 configuration classes
- **Configuration Reference**: [`dapr_agents/agents/configs.py`](../../dapr_agents/agents/configs.py) - Source code
- **Simple Agent**: [`03-durable-agent-tool-call`](../03-durable-agent-tool-call/) - Basic agent without configs
- **Dapr Workflows**: [Dapr Workflow Documentation](https://docs.dapr.io/developing-applications/building-blocks/workflow/)

