# MCP Agent with Dapr Workflows

This demo shows how to run an AI agent inside a Dapr Workflow, calling tools exposed via the [Model Context Protoco (MCP)](https://modelcontextprotocol.io/introduction).

Unlike the lightweight notebook-based examples, this setup runs a full Dapr agent using:

✅ Durable task orchestration with Dapr Workflows
✅ Tools served via MCP (stdio or SSE)
✅ Full integration with the Dapr ecosystem

## 🛠️ Project Structure

```text
.
├── app.py           # Main entrypoint: runs a Dapr Agent and workflow on port 8001
├── tools.py         # MCP tool definitions (get_weather, jump)
├── server.py        # Starlette-based SSE server
|-- client.py        # Script to send an HTTP request to the Agent over port 8001
├── components/      # Dapr pubsub + state components (Redis, etc.)
├── requirements.txt
└── README.md
```

## 📦 Installation

Install dependencies:

```python
pip install -r requirements.txt
```

Make sure you have Dapr installed and initialized:

```bash
dapr init
```

## 🧰 MCP Tool Server

Your agent will call tools defined in tools.py, served via FastMCP:

```python
@mcp.tool()
async def get_weather(location: str) -> str:
    ...

@mcp.tool()
async def jump(distance: str) -> str:
    ...
```

These tools can be served in one of two modes:

### STDIO Mode (local execution)

No external server needed — the agent runs the MCP server in-process.

✅ Best for internal experiments or testing
🚫 Not supported for agents that rely on external workflows (e.g., Dapr orchestration)

### SSE Mode (recommended for Dapr workflows)

In this demo, we run the MCP server as a separate Starlette + Uvicorn app:

```python
python server.py --server_type sse --host 127.0.0.1 --port 8000
```

This exposes:

* `/sse` for the SSE stream
* `/messages/` for tool execution

Used by the Dapr agent in this repo.

## 🚀 Running the Dapr Agent

Start the MCP server in SSE mode:

```python
python server.py --server_type sse --port 8000
```

Then in a separate terminal, run the agent workflow:

```bash
dapr run --app-id weatherappmcp --resources-path components/ -- python app.py
```

Once agent is ready, run the `client.py` script to send a message to it. 

```bash
python3 client.py
```

You will see the state of the agent in a json file in the same directory.

```
{
    "instances": {
        "e098e5b85d544c84a26250be80316152": {
            "input": "What is the weather in New York?",
            "output": "The current temperature in New York, USA, is 66\u00b0F.",
            "start_time": "2025-04-05T05:37:50.496005",
            "end_time": "2025-04-05T05:37:52.501630",
            "messages": [
                {
                    "id": "e8ccc9d2-1674-47cc-afd2-8e68b91ff791",
                    "role": "user",
                    "content": "What is the weather in New York?",
                    "timestamp": "2025-04-05T05:37:50.516572",
                    "name": null
                },
                {
                    "id": "47b8db93-558c-46ed-80bb-8cb599c4272b",
                    "role": "assistant",
                    "content": "The current temperature in New York, USA, is 66\u00b0F.",
                    "timestamp": "2025-04-05T05:37:52.499945",
                    "name": null
                }
            ],
            "last_message": {
                "id": "47b8db93-558c-46ed-80bb-8cb599c4272b",
                "role": "assistant",
                "content": "The current temperature in New York, USA, is 66\u00b0F.",
                "timestamp": "2025-04-05T05:37:52.499945",
                "name": null
            },
            "tool_history": [
                {
                    "content": "New York, USA: 66F.",
                    "role": "tool",
                    "tool_call_id": "call_LTDMHvt05e1tvbWBe0kVvnUM",
                    "id": "2c1535fe-c43a-42c1-be7e-25c71b43c32e",
                    "function_name": "LocalGetWeather",
                    "function_args": "{\"location\":\"New York, USA\"}",
                    "timestamp": "2025-04-05T05:37:51.609087"
                }
            ],
            "source": null,
            "source_workflow_instance_id": null
        }
    }
}
```

