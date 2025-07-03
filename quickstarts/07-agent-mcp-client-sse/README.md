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

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

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

Create the agent that connects to these tools in `app.py` over MCP with SSE transport:

```python
# Load MCP tools from server using SSE
client = MCPClient()
await client.connect_sse("local", url="http://localhost:8000/sse")
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

1. Start the MCP server in SSE mode:

```bash
python server.py --server_type sse --port 8000
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

## Next Steps

After completing this quickstart, you might want to explore:
- Creating more complex MCP tools with actual API integrations
- Deploying your agent as a Dapr microservice in Kubernetes
- Exploring the [MCP specification](https://modelcontextprotocol.io/) for advanced usage