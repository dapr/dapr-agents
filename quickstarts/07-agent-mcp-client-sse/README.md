# MCP Agent with SSE Transport

This quickstart demonstrates how to build a simple agent that uses tools exposed via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) over SSE (Server-Sent Events) transport. You'll learn how to create MCP tools in a standalone server and connect to them using SSE communication.

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

The quickstart includes an OpenAI component configuration in the `components` directory. You have two options to configure your API key:

### Option 1: Using Environment Variables (Recommended)

1. Create a `.env` file in the project root and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

2. When running the examples with Dapr, use the helper script to resolve environment variables:
```bash
# Get the environment variables from the .env file:
export $(grep -v '^#' ../../.env | xargs)

# Create a temporary resources folder with resolved environment variables
temp_resources_folder=$(../resolve_env_templates.py ./components)

# Run your dapr command with the temporary resources
dapr run --app-id weatherappmcp --dapr-http-port 3500 --resources-path $temp_resources_folder -- python app.py

# Clean up when done
rm -rf $temp_resources_folder
```

Note: The temporary resources folder will be automatically deleted when the Dapr sidecar is stopped or when the computer is restarted.

### Option 2: Direct Component Configuration

You can directly update the `key` in [components/openai.yaml](components/openai.yaml):
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

Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.

Note: Many LLM providers are compatible with OpenAI's API (DeepSeek, Google AI, etc.) and can be used with this component by configuring the appropriate parameters. Dapr also has [native support](https://docs.dapr.io/reference/components-reference/supported-conversation/) for other providers like Google AI, Anthropic, Mistral, DeepSeek, etc.

## Examples

### MCP Tool Creation

First, create MCP tools in `tools.py`:

```python
mcp = FastMCP("TestServer")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather information for a specific location."""
    temperature = random.randint(60, 80)
    return f"{location}: {temperature}F."
```

### SSE Server Creation

Set up the SSE server for your MCP tools in `server.py`.

### Agent Creation

`app.py` now mirrors the durable service pattern used in other quickstarts:

1. Connect to the MCP server via SSE and load tools.
2. Build a `DurableAgent` with Dapr configs (`AgentPubSubConfig`, state, registry, memory, execution).
3. Start the agent and host it via `AgentRunner().serve(agent, port=8001)`.

```python
tools = asyncio.run(_load_mcp_tools())

agent = DurableAgent(
    name="Stevie",
    role="Weather Assistant",
    goal="Help humans get weather, travel, and location details using smart tools.",
    instructions=[...],
    tools=tools,
    pubsub=pubsub_config,
    registry=registry_config,
    execution=execution_config,
    memory=memory_config,
    state=state_config,
)
agent.start()

runner = AgentRunner()
runner.serve(agent, port=8001)
```

### Running the Example

1. Start the MCP server in SSE mode:

```bash
python server.py --server_type sse --port 8000
```

2. In a separate terminal window, start the agent with Dapr (ensure components are rendered first):

```bash
dapr run \
  --app-id weatherappmcp \
  --app-port 8001 \
  --resources-path $temp_resources_folder \
  -- python app.py
rm -rf $temp_resources_folder
```

3. Send a test request to the agent:

```bash
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the weather in New York?"}'
```

**Expected output:** The agent will initialize the MCP client, connect to the tools module via SSE transport, and fetch weather information for New York using the MCP tools. The results will be stored in state files.

## Key Concepts

### MCP Tool Definition
- The `@mcp.tool()` decorator registers functions as MCP tools
- Each tool has a docstring that helps the LLM understand its purpose

### SSE Transport
- SSE (Server-Sent Events) transport enables network-based communication
- Perfect for distributed setups where tools run as separate services
- Allows multiple agents to connect to the same tool server

### Dapr Integration
- The `DurableAgent` class creates a service that runs inside a Dapr workflow
- Dapr components (pubsub, state stores) manage message routing and state persistence
- The agent's conversation history and tool calls are saved in Dapr state stores

### Execution Flow
1. MCP server starts with tools exposed via SSE endpoint
2. Agent connects to the MCP server via SSE
3. The agent receives a user query via HTTP
4. The LLM determines which MCP tool to use
5. The agent sends the tool call to the MCP server
6. The server executes the tool and returns the result
7. The agent formulates a response based on the tool result
8. State is saved in the configured Dapr state store

## Alternative: Using STDIO Transport

While this quickstart uses SSE transport, MCP also supports STDIO for process-based communication. This approach is useful when:

- Tools need to run in the same process as the agent
- Simplicity is preferred over network distribution
- You're developing locally and don't need separate services

To explore STDIO transport, check out the related [MCP with STDIO Transport quickstart](../07-agent-mcp-client-stdio).

## Troubleshooting

1. **OpenAI API Key**: Ensure your key is correctly set in the `.env` file
2. **Server Connection**: If you see SSE connection errors, make sure the server is running on the correct port
3. **Dapr Setup**: Verify that Dapr is installed and that Redis is running for state stores
4. **Module Import Errors**: Verify that all dependencies are installed correctly
5. **gRPC Timeout**: For longer LLM responses set `DAPR_API_TIMEOUT_SECONDS=300` so the Dapr client waits beyond the 60 s default.

## Next Steps

After completing this quickstart, you might want to explore:
- Creating more complex MCP tools with actual API integrations
- Deploying your agent as a Dapr microservice in Kubernetes
- Exploring the [MCP specification](https://modelcontextprotocol.io/) for advanced usage
