# Hello World with Dapr Agents

This quickstart provides a hands-on introduction to Dapr Agents through simple examples. You'll learn the fundamentals of working with LLMs, creating basic agents, implementing the ReAct pattern, and setting up simple workflows - all in less than 20 lines of code per example.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key

## Environment Setup

### Option 1: Using pip (Traditional)

```bash
# Create a virtual environment
python3.10 -m venv .venv

# Activate the virtual environment 
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip-compile pyproject.toml
pip install -r requirements.txt

# For vectorstore functionality (optional)
pip install sentence-transformers chromadb 'posthog<6.0.0'
```

### Option 2: Using uv (Recommended)

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install core dependencies
uv pip install -r requirements.txt

# For vectorstore functionality (optional)
uv add sentence-transformers chromadb 'posthog<6.0.0'
```

### Option 3: Install with extras (uv only)

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install with vectorstore extras
uv pip install -e ".[vectorstore]"
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

## Examples

### 1. Basic LLM Usage

Run the basic LLM example to see how to interact with OpenAI's language models:

<!-- STEP
name: Run basic LLM example
expected_stdout_lines:
  - "Got response:"
timeout_seconds: 30
output_match_mode: substring
-->
```bash
python 01_ask_llm.py
```
<!-- END_STEP -->

This example demonstrates the simplest way to use Dapr Agents' OpenAIChatClient:

```python
from dapr_agents import OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()
llm = OpenAIChatClient()
response = llm.generate("Tell me a joke")
print("Got response:", response.get_content())
```

**Expected output:** The LLM will respond with a joke.

### 2. Simple Agent with Tools

Run the agent example to see how to create an agent with custom tools:

<!-- STEP
name: Run simple agent with tools example
expected_stdout_lines:
  - "user:"
  - "What's the weather?"
  - "assistant:"
  - "Function name: MyWeatherFunc"
  - "MyWeatherFunc(tool)"
  - "It's 72°F and sunny"
  - "assistant:"
  - "The current weather is 72°F and sunny."
timeout_seconds: 30
output_match_mode: substring
-->
```bash
python 02_build_agent.py
```
<!-- END_STEP -->

This example shows how to create a basic agent with a custom tool:

```python
import asyncio
from dapr_agents import tool, Agent
from dotenv import load_dotenv

load_dotenv()

@tool
def my_weather_func() -> str:
    """Get current weather."""
    return "It's 72°F and sunny"

async def main():
    weather_agent = Agent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[my_weather_func]
    )

    response = await weather_agent.run("What's the weather?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

**Expected output:** The agent will use the weather tool to provide the current weather.

### 3. Durable Agent

A stateful agent that uses Dapr Workflows to ensure durability and persistence of agent reasoning.

Make sure Dapr is initialized on your system:

```bash
dapr init
```

Run the assistant agent example to see how to create a stateful agent with persistent memory:

```bash
dapr run --app-id stateful-llm --app-port 8001 --dapr-http-port 3500 --resources-path components/ -- python 03_durable_agent.py
```

This example demonstrates a stateful travel planning assistant that:
1. Remembers user context persistently (across restarts)
2. Uses a tool to search for flight options
3. Exposes a REST API for workflow interaction
4. Stores execution state in Dapr workflow state stores

```python
#!/usr/bin/env python3
"""
Stateful Augmented LLM Pattern demonstrates:
1. Memory - remembering user preferences
2. Tool use - accessing external data
3. LLM abstraction
4. Durable execution of tools as workflow actions
"""
import asyncio
import logging
from typing import List
from pydantic import BaseModel, Field
from dapr_agents import tool, DurableAgent
from dapr_agents.memory import ConversationDaprStateMemory
from dotenv import load_dotenv

# Define tool output model
class FlightOption(BaseModel):
    airline: str = Field(description="Airline name")
    price: float = Field(description="Price in USD")

# Define tool input model
class DestinationSchema(BaseModel):
    destination: str = Field(description="Destination city name")

# Define flight search tool
@tool(args_model=DestinationSchema)
def search_flights(destination: str) -> List[FlightOption]:
    """Search for flights to the specified destination."""
    # Mock flight data (would be an external API call in a real app)
    return [
        FlightOption(airline="SkyHighAir", price=450.00),
        FlightOption(airline="GlobalWings", price=375.50)
    ]

async def main():
    try:
        # Initialize TravelBuddy agent
        travel_planner = DurableAgent(
            name="TravelBuddy",
            role="Travel Planner",
            goal="Help users find flights and remember preferences",
            instructions=[
                "Find flights to destinations",
                "Remember user preferences",
                "Provide clear flight info"
            ],
            tools=[search_flights],
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="workflow_state",
            agents_registry_store_name="registrystatestore",
            agents_registry_key="agents_registry",
            memory=ConversationDaprStateMemory(
                store_name="conversationstore", session_id="my-unique-id"
            )
        )

        travel_planner.as_service(port=8001)
        await travel_planner.start()
        print("Travel Planner Agent is running")

    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

### Interacting with the Agent

Unlike simpler agents, this stateful agent exposes a REST API for workflow interactions:

#### Start a new workflow:

```bash
curl -i -X POST http://localhost:8001/start-workflow \
  -H "Content-Type: application/json" \
  -d '{"task": "I want to find flights to Paris"}'
```

You'll receive a workflow ID in response, which you can use to track progress.

#### Check workflow status:

```bash
# Replace WORKFLOW_ID with the ID from the previous response
curl -i -X GET http://localhost:3500/v1.0/workflows/dapr/WORKFLOW_ID
```

### How It Works

The key components of this implementation are:

1. **Persistent Memory**: The agent stores conversation state in Dapr's state store, enabling it to remember context across sessions and system restarts.

2. **Workflow Orchestration**: Long-running tasks are managed through Dapr's workflow system, providing:
    - Durability - workflows survive process crashes
    - Observability - track status and progress
    - Recoverability - automatic retry on failures

3. **Tool Integration**: A flight search tool is defined using the `@tool` decorator, which automatically handles input validation and type conversion.

4. **Service Exposure**: The agent exposes REST endpoints to start and manage workflows.

### 4. Simple Workflow

Run the workflow example to see how to create a multi-step LLM process:

<!-- STEP
name: Run a simple workflow example
expected_stdout_lines:
  - "Outline:"
  - "Blog post:"
  - "Result:"
output_match_mode: substring
-->
```bash
dapr run --app-id dapr-agent-wf -- python 04_chain_tasks.py
```
<!-- END_STEP -->

This example demonstrates how to create a workflow with multiple tasks:

```python
from dapr_agents.workflow import WorkflowApp, workflow, task
from dapr.ext.workflow import DaprWorkflowContext

from dotenv import load_dotenv

load_dotenv()

@workflow(name='analyze_topic')
def analyze_topic(ctx: DaprWorkflowContext, topic: str):
    # Each step is durable and can be retried
    outline = yield ctx.call_activity(create_outline, input=topic)
    blog_post = yield ctx.call_activity(write_blog, input=outline)
    return blog_post

@task(description="Create a detailed outline about {topic}")
def create_outline(topic: str) -> str:
    pass

@task(description="Write a comprehensive blog post following this outline: {outline}")
def write_blog(outline: str) -> str:
    pass

if __name__ == '__main__':
    wfapp = WorkflowApp()

    results = wfapp.run_and_monitor_workflow_sync(
        analyze_topic,
        input="AI Agents"
    )
    print(f"Result: {results}")
```

**Expected output:** The workflow will create an outline about AI Agents and then generate a blog post based on that outline.

### 5. Agent with Vector Store

**Prerequisites:** This example requires vectorstore dependencies. Install them using one of these methods:

**Using uv (recommended):**
```bash
uv add sentence-transformers chromadb 'posthog<6.0.0'
```

**Using pip:**
```bash
pip install sentence-transformers chromadb 'posthog<6.0.0'
```

**Or install with extras (uv only):**
```bash
uv pip install -e ".[vectorstore]"
```

Run the vector store agent example to see how to create an agent that can search and store documents:

<!-- STEP
name: Run agent with vector store example
expected_stdout_lines:
  - "Starting Vector Database Agent..."
  - "Add Document Response:"
  - "Search Response:"
output_match_mode: substring
-->
```bash
python 05_agent_with_vectorstore.py
```
<!-- END_STEP -->

This example demonstrates how to create an agent with vector store capabilities:

```python
from dapr_agents import Agent
from dapr_agents.tool import tool
from dapr_agents.storage.vectorstores import ChromaVectorStore
from dapr_agents.document.embedder.sentence import SentenceTransformerEmbedder

@tool
def search_documents(query: str) -> str:
    """Search for documents in the vector store"""
    return f"Found documents related to '{query}': Document 1, Document 2"

@tool
def add_document(content: str, metadata: str = "") -> str:
    """Add a document to the vector store"""
    return f"Added document: {content[:50]}..."

async def main():
    embedding_function = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    
    vector_store = ChromaVectorStore(
        name="demo_vectorstore",
        embedding_function=embedding_function,
        persistent=True,
        path="./chroma_db",
    )

    agent = Agent(
        name="VectorBot",
        role="Vector Database Assistant",
        goal="Help with document search and storage",
        instructions=[
            "Search documents in vector store",
            "Add documents to vector store",
            "Provide relevant information from stored documents",
        ],
        tools=[search_documents, add_document],
        vector_store=vector_store,
    )

    response = await agent.run("Add a document about machine learning basics")
    print("Add Document Response:")
    print(response)
    
    response = await agent.run("Search for documents about machine learning")
    print("Search Response:")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

## Key Concepts

- **OpenAIChatClient**: The interface for interacting with OpenAI's LLMs
- **Agent**: A class that combines an LLM with tools and instructions
- **@tool decorator**: A way to create tools that agents can use
- **DurableAgent**: An agent that follows the Reasoning + Action pattern and achieves durability through Dapr Workflows
- **VectorStore**: Persistent storage for document embeddings that enables semantic search capabilities
- **WorkflowApp**: A Dapr-powered way to create stateful, multi-step processes

## Dapr Integration

These examples don't directly expose Dapr building blocks, but they're built on Dapr Agents which behind the scenes leverages the full capabilities of the Dapr runtime:

- **Resilience**: Built-in retry policies, circuit breaking, and timeout handling external systems interactions
- **Orchestration**: Stateful, durable workflows that can survive process restarts and continue execution from where they left off
- **Interoperability**: Pluggable component architecture that works with various backends and cloud services without changing application code
- **Scalability**: Distribute agents across infrastructure, from local development to multi-node Kubernetes clusters
- **Event-Driven**: Pub/Sub messaging for event-driven agent collaboration and coordination
- **Observability**: Integrated distributed tracing, metrics collection, and logging for visibility into agent operations
- **Security**: Protection through scoping, encryption, secret management, and authentication/authorization controls

In the later quickstarts, you'll see explicit Dapr integration through state stores, pub/sub, and workflow services.

## Troubleshooting

1. **API Key Issues**: If you see an authentication error, verify your OpenAI API key in the `.env` file
2. **Python Version**: If you encounter compatibility issues, make sure you're using Python 3.10+
3. **Environment Activation**: Ensure your virtual environment is activated before running examples
4. **Import Errors**: If you see module not found errors, verify that `pip install -r requirements.txt` completed successfully

## Next Steps

After completing these examples, move on to the [LLM Call quickstart](../02_llm_call_open_ai/README.md) to learn more about structured outputs from LLMs.