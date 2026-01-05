# MCP Agent with Streamable HTTP transport

This quickstart demonstrates how to build a simple agent that uses tools exposed via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) over Streamable HTTP transport. You'll learn how to create MCP tools in a standalone server and connect to them using Streamable HTTP communication.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key
- Dapr CLI and Docker (for Redis/pub-sub components)

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

The quickstart ships with component templates under `components/`. They include:

- `openai.yaml` for LLM access
- Redis-backed state stores (`agentstatestore`, `agentregistrystore`, `conversationstore`, `workflowstatestore`)
- `messagepubsub.yaml` for pub/sub triggers

### Option 1: Using Environment Variables (Recommended)

1. Create a `.env` file and add your OpenAI key:
```env
OPENAI_API_KEY=your_api_key_here
```

2. Render the templates before running Dapr:
```bash
export $(grep -v '^#' ../../.env | xargs)
temp_resources_folder=$(../resolve_env_templates.py ./components)

dapr run \
  --app-id weatherappmcp-http \
  --app-port 8001 \
  --resources-path "$temp_resources_folder" \
  -- python app.py

rm -rf "$temp_resources_folder"
```

### Option 2: Direct Component Configuration

Edit `components/openai.yaml` and replace the `value` for `name: key` with your API key. Avoid committing secrets back to source control.

### Additional Components

Run `dapr init` so Redis and the sidecar are available. The provided YAML files already set `actorStateStore: "true"` for the workflow state store.

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

`app.py` now mirrors the other MCP quickstarts:

1. Connect to the MCP server via streamable HTTP, fetch tools, and close the client.
2. Configure a `DurableAgent` with pub/sub, memory, registry, and workflow state stores.
3. Host the agent with `AgentRunner.serve(agent, port=8001)` so `/run` is exposed automatically.

### Running the Example

1. Start the MCP server in Streamable HTTP mode:

```bash
python server.py --server_type streamable-http --port 8000
```

2. In another terminal, render the components and start the agent:

```bash
temp_resources_folder=$(../resolve_env_templates.py ./components)
dapr run \
  --app-id weatherappmcp-http \
  --app-port 8001 \
  --resources-path "$temp_resources_folder" \
  -- python app.py
rm -rf "$temp_resources_folder"
```

3. Send a test request to the agent:

```bash
curl -X POST http://localhost:8001/run \
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

1. **OpenAI API Key**: Ensure your key is present in `.env` or baked into `openai.yaml`.
2. **Server Connection**: If the agent can’t load tools, confirm `server.py` is running in streamable HTTP mode on the correct port.
3. **Dapr Setup**: Redis must be running (installed via `dapr init`). If you see `state store ... is not found`, ensure the rendered components were passed to `dapr run`.
4. **gRPC Deadline**: For longer prompts/responses set `DAPR_API_TIMEOUT_SECONDS=300` so the Dapr client waits beyond the 60 s default.
5. **Dependencies**: Run `pip install -r requirements.txt` to install `dapr-agents`, `starlette`, and other requirements.

## Next Steps

After completing this quickstart, you might want to explore:
- Creating more complex MCP tools with actual API integrations
- Deploying your agent as a Dapr microservice in Kubernetes
- Exploring the [MCP specification](https://modelcontextprotocol.io/) for advanced usage
