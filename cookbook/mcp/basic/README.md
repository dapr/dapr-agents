# 🧪 Basic MCP Agent Playground

This demo shows how to use a **lightweight agent** to call tools served via the [Model Context Protoco (MCP)](https://modelcontextprotocol.io/introduction). The agent uses a simple pattern from `dapr_agents` — but **without running inside Dapr**.

It’s a minimal, Python-based setup for:

- Exploring how MCP tools work
- Testing stdio and SSE transport
- Running tool-calling agents
- Experimenting **without** durable workflows or Dapr dependencies

> 🧠 Looking for something more robust?  
> Check out the full `dapr_agents` repo to see how we run these agents inside Dapr workflows with durable task orchestration and state management.

---

## 🛠️ Project Structure

```text
.
├── tools.py         # Registers two tools via FastMCP
├── server.py        # Starts the MCP server in stdio or SSE mode
├── stdio.ipynb    # Example using ToolCallingAgent over stdio
├── sse.ipynb      # Example using ToolCallingAgent over SSE
├── requirements.txt
└── README.md
```

## Installation

Before running anything, make sure to install the dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Starting the MCP Tool Server

The server exposes two tools via MCP:

* `get_weather(location: str) → str`
* `jump(distance: str) → str`

Defined in `tools.py`, these tools are registered using FastMCP.

You can run the server in two modes:

### ▶️ 1. STDIO Mode

This runs inside the notebook. It's useful for quick tests because the MCP server doesn't need to be running in a separate terminal.

* This is used in `stdio.ipynb`
* The agent communicates with the tool server via stdio transport

### 🌐 2. SSE Mode (Starlette + Uvicorn)
This mode requires running the server outside the notebook (in a terminal).

```python
python server.py --server_type sse --host 127.0.0.1 --port 8000
```

The server exposes:

* `/sse` for the SSE connection
* `/messages/` to receive tool calls

Used by `sse.ipynb`

📌 You can change the port and host using --host and --port.

## 📓 Notebooks
There are two notebooks in this repo that show basic agent behavior using MCP tools:

| Notebook | Description | Transport |
| --- | --- | --- |
| stdio.ipynb | Uses ToolCallingAgent via mcp.run("stdio") | STDIO |
| sse.ipynb	Uses | ToolCallingAgent with SSE tool server | SSE |

Each notebook runs a basic `ToolCallingAgent`, using tools served via MCP. These agents are not managed via Dapr or durable workflows — it's pure Python execution with async support.

## 🔄 What’s Next?

After testing these lightweight agents, you can try:

* Running the full dapr_agents workflow system
* Registering more complex MCP tools
* Using other agent types (Agent or DurableAgent)
* Testing stateful, durable workflows using Dapr + MCP tools