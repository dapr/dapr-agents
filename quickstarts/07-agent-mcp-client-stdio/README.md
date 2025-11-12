# MCP Agent with STDIO Transport

This quickstart demonstrates how to build a simple agent that uses tools exposed via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) over STDIO transport. You'll learn how to create MCP tools in a standalone module and connect to them using STDIO communication.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key
- Dapr CLI and Docker (for the state stores and pub/sub sidecars)

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

The quickstart ships with Dapr component templates under `components/`. You can inject your OpenAI key via environment variables or by editing the component directly.

### Option 1: Using Environment Variables (Recommended)

1. Create a `.env` file in the project root and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

2. When running the example, resolve the templates before starting Dapr:
```bash
# Load env vars from the repo-level .env if present
export $(grep -v '^#' ../../.env | xargs)

# Render component templates with real secrets
temp_resources_folder=$(../resolve_env_templates.py ./components)

dapr run \
  --app-id weatherappmcp-stdio \
  --resources-path "$temp_resources_folder" \
  -- python agent.py

rm -rf "$temp_resources_folder"
```

### Option 2: Direct Component Configuration

Update `components/openai.yaml` by replacing the `value` for `name: key` with your API key. This approach is convenient for demos but avoid committing secrets to source control.

### Additional Components

Make sure Dapr is initialized:
```bash
dapr init
```
The `components/` folder already includes Redis-backed state stores (`agentstatestore`, `agentregistrystore`, `conversationstore`), pub/sub, and workflow state components required by the durable agent.

## MCP Tools

`tools.py` defines the MCP tools using `FastMCP`. The STDIO transport starts this module as a subprocess automatically—no extra server process is required:

```python
mcp = FastMCP("TestServer")

@mcp.tool()
async def get_weather(location: str) -> str:
    temperature = random.randint(60, 80)
    return f"{location}: {temperature}F."
```

## Running the Example

1. Render the components and start the agent (as shown above).
2. In another terminal, send a request to the hosted endpoint:
```bash
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the weather in New York?"}'
```

The agent connects to the MCP tools over STDIO, reasons about the query, invokes `get_weather`, and returns the final response. Conversation history and tool traces are persisted in the configured Dapr state stores.

## Key Concepts

### MCP Tool Definition
- The `@mcp.tool()` decorator registers functions as MCP tools
- Each tool has a docstring that helps the LLM understand its purpose

### STDIO Transport
- STDIO transport uses standard input/output streams for communication
- No network ports or HTTP servers are required for this transport
- Ideal for local development and testing

### Agent Setup with MCP Client
- The `MCPClient` class manages connections to MCP tool servers
- `connect_stdio()` starts a subprocess and establishes communication
- The client translates MCP tools into agent tools automatically

### Execution Flow
1. The agent renders Dapr components and starts the Workflow runtime.
2. `MCPClient.connect_stdio()` launches `tools.py` as a subprocess and discovers available tools.
3. An HTTP request to `/run` triggers the durable workflow (via `AgentRunner.serve`).
4. The LLM decides whether to call an MCP tool.
5. Tool calls are dispatched over STDIO; results flow back into the conversation loop.
6. Final responses are returned to the HTTP caller and saved in Dapr state.

## Alternative: Using SSE Transport

While this quickstart uses STDIO transport, MCP also supports Server-Sent Events (SSE) for network-based communication. This approach is useful when:

- Tools need to run as separate services
- Tools are distributed across different machines
- You need long-running services that multiple agents can connect to

To explore SSE transport, check out the related [MCP with SSE Transport quickstart](../07-agent-mcp-client-sse).

## Troubleshooting

1. **OpenAI API Key**: Ensure the key is present in `.env` or baked into `openai.yaml`.
2. **STDIO communication**: If the agent hangs while loading tools, run `python tools.py` manually to confirm it starts.
3. **Dapr Components**: Redis must be running (provided by `dapr init`). Check the sidecar logs if you see `state store ... is not found`.
4. **gRPC Deadline**: For long prompts/responses set `DAPR_API_TIMEOUT_SECONDS=300` so the Dapr client waits beyond the 60 s default.
5. **Module Imports**: Run `pip install -r requirements.txt` to install `dapr-agents`, `fastmcp`, and other dependencies.

## Next Steps

After completing this quickstart, you might want to explore:
- Checkout SSE transport example [MCP with SSE Transport quickstart](../07-agent-mcp-client-sse).
- Exploring the [MCP specification](https://modelcontextprotocol.io/) for advanced usage 
