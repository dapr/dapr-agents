# MCP Agent with Streamable HTTP transport

This quickstart demonstrates how to build a simple agent that uses tools exposed via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) over Streamable HTTP transport. You'll learn how to create MCP tools in a standalone server and connect to them using Streamable HTTP communication.

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
dapr run --app-id weatherappmcp --app-port 8001 --dapr-http-port 3500 --resources-path $temp_resources_folder -- python app.py

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

### Streamable HTTP Server Creation

Set up the server for your MCP tools in `server.py`.

### Agent Creation

Create the agent that connects to these tools in `app.py` over MCP with Streamable HTTP transport:

```python
# Load MCP tools from server using Streamable HTTP transport
client = MCPClient()
await client.connect_streamable_http(server_name="local", url="http://localhost:8000/mcp/")
tools = client.get_all_tools()

# Create the Weather Agent using MCP tools
weather_agent = DurableAgent(
    role="Weather Assistant",
    name="Stevie",
    goal="Help humans get weather and location info using smart tools.",
    instructions=["Instrictions go here"],
    tools=tools,
    message_bus_name="messagepubsub",
    state_store_name="workflowstatestore",
    state_key="workflow_state",
    agents_registry_store_name="agentstatestore",
    agents_registry_key="agents_registry",
).as_service(port=8001)
 
```

### Running the Example

1. Start the MCP server in Streamable HTTP mode:

```bash
python server.py --server_type streamable-http --port 8000
```

2. In a separate terminal window, start the agent with Dapr:

```bash
dapr run --app-id weatherappmcp --app-port 8001 --dapr-http-port 3500 --resources-path ./components/ -- python app.py
```

3. Send a test request to the agent:

```bash
curl -X POST http://localhost:8001/start-workflow \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the weather in New York?"}'
```

**Expected output:** The agent will initialize the MCP client, connect to the tools module via Streamable HTTP transport, and fetch weather information for New York using the MCP tools. The results will be stored in state files.

## Key Concepts

### MCP Tool Definition
- The `@mcp.tool()` decorator registers functions as MCP tools
- Each tool has a docstring that helps the LLM understand its purpose

### Streamable HTTP Transport
- Streamable HTTP is a modern, robust transport in MCP that uses standard HTTP(S) for both requests and real-time streaming responses.
- Ideal for cloud-native and distributed deployments—agents and tools can communicate across networks, containers, or Kubernetes clusters.
- Enables advanced features like resumable sessions, stateless operation, and efficient streaming of large or incremental results.
- Multiple agents and clients can connect to the same tool server, each with isolated or shared sessions as needed.

### Dapr Integration
- The `DurableAgent` class creates a service that runs inside a Dapr workflow
- Dapr components (pubsub, state stores) manage message routing and state persistence
- The agent's conversation history and tool calls are saved in Dapr state stores

### Execution Flow
1. MCP server starts with tools exposed via Streamable HTTP endpoint
2. Agent connects to the MCP server via Streamable HTTP transport
3. The agent receives a user query via HTTP
4. The LLM determines which MCP tool to use
5. The agent sends the tool call to the MCP server
6. The server executes the tool and returns the result
7. The agent formulates a response based on the tool result
8. State is saved in the configured Dapr state store

## Troubleshooting

1. **OpenAI API Key**: Ensure your key is correctly set in the `.env` file
2. **Server Connection**: If you see  connection errors, make sure the server is running on the correct port
3. **Dapr Setup**: Verify that Dapr is installed and that Redis is running for state stores
4. **Module Import Errors**: Verify that all dependencies are installed correctly

## Next Steps

After completing this quickstart, you might want to explore:
- Creating more complex MCP tools with actual API integrations
- Deploying your agent as a Dapr microservice in Kubernetes
- Exploring the [MCP specification](https://modelcontextprotocol.io/) for advanced usage