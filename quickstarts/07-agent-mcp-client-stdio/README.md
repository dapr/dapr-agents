# MCP Agent with STDIO Transport

This quickstart demonstrates how to build a simple agent that uses tools exposed via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) over STDIO transport. You'll learn how to create MCP tools in a standalone module and connect to them using STDIO communication.

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

### Agent Creation

Then, create the agent that connects to these tools in `agent.py` over MCP:

```python
client = MCPClient()

# Connect to MCP server using STDIO transport
await client.connect_stdio(
    server_name="local",
    command=sys.executable,  # Use the current Python interpreter
    args=["tools.py"]  # Run tools.py directly
)

# Get available tools from the MCP instance
tools = client.get_all_tools()

# Create the Weather Agent using MCP tools
weather_agent = Agent(
    name="Stevie",
    role="Weather Assistant",
    goal="Help humans get weather and location info using MCP tools.",
    instructions=["Instrictions go here"],
    tools=tools,
)    
```

### Running the Example

Run the agent script:

```bash
python agent.py
```

**Expected output:** The agent will initialize the MCP client, connect to the tools module via STDIO, and fetch weather information for New York using the MCP tools.

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
1. Agent starts the tools module as a subprocess
2. MCPClient connects to the subprocess via STDIO
3. The agent receives a user query
4. The LLM determines which MCP tool to use
5. The agent sends the tool call to the tools subprocess
6. The subprocess executes the tool and returns the result
7. The agent formulates a response based on the tool result

## Alternative: Using SSE Transport

While this quickstart uses STDIO transport, MCP also supports Server-Sent Events (SSE) for network-based communication. This approach is useful when:

- Tools need to run as separate services
- Tools are distributed across different machines
- You need long-running services that multiple agents can connect to

To explore SSE transport, check out the related [MCP with SSE Transport quickstart](../07-agent-mcp-client-sse).

## Troubleshooting

1. **OpenAI API Key**: Ensure your key is correctly set in the `.env` file
2. **Subprocess Communication**: If you see STDIO errors, make sure tools.py can run independently
3. **Module Import Errors**: Verify that all dependencies are installed correctly

## Next Steps

After completing this quickstart, you might want to explore:
- Checkout SSE transport example [MCP with SSE Transport quickstart](../07-agent-mcp-client-sse).
- Exploring the [MCP specification](https://modelcontextprotocol.io/) for advanced usage 