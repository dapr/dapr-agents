# Hello World with Dapr Agents

This quickstart provides a hands-on introduction to Dapr Agents through simple examples. You'll learn the fundamentals of working with LLMs, creating basic agents, implementing the ReAct pattern, and setting up simple workflows - all in less than 20 lines of code per example.

## Prerequisites

- Python 3.10+ (recommended)
- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
- OpenAI API key (you can put in an .env file or directly in the `openai.yaml` file, but we recommend the .env file that is gitignored)

## Environment Setup

<details open>
<summary><strong>Option 1: Using uv (Recommended)</strong></summary>

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install core dependencies
uv pip install -r requirements.txt
```

</details>

<details>
<summary><strong>Option 2: Using pip</strong></summary>

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

</details>


## OpenAI Configuration

> **Warning**
> The examples will not work if you do not have an OpenAI API key configured.

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
dapr run --resources-path $temp_resources_folder -- python your_script.py

# Clean up when done
rm -rf $temp_resources_folder
```

Note: The temporary resources folder will be automatically deleted when the Dapr sidecar is stopped or when the computer is restarted.

### Option 2: Direct Component Configuration

You can directly update the `key` in [components/openai.yaml](components/openai.yaml):
```yaml
metadata:
  - name: key
    value: "YOUR_OPENAI_API_KEY"
```

Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.

Note: Many LLM providers are compatible with OpenAI's API (DeepSeek, Google AI, etc.) and can be used with this component by configuring the appropriate parameters. Dapr also has [native support](https://docs.dapr.io/reference/components-reference/supported-conversation/) for other providers like Google AI, Anthropic, Mistral, DeepSeek, etc.

## Examples

### 1. Basic LLM Usage

Run the basic LLM example to see how to interact with OpenAI's language models:

```bash
python 01_ask_llm.py
```

This example demonstrates the simplest way to use Dapr Agents' OpenAIChatClient.

**Expected output:** The LLM will respond with a joke.

### 2. Simple Agent with Tools

Run the agent example to see how to create an agent with custom tools.
```bash
python 02_build_agent.py
```

This example shows how to create a basic agent with a custom tool.

**Expected output:** The agent will use the weather tool to provide the current weather.

### 3. Durable Agent

This example turns the TravelBuddy agent into a durable workflow-backed service. It uses the Dapr ChatClient (configured through `components/openai.yaml`) so remember to run `resolve_env_templates.py` and provide your `{YOUR_OPENAI_API_KEY}` before starting. Ensure `dapr init` has been run on your machine.

Choose one of the following entry points depending on how you want to host the agent:

| Script | Mode | When to use it |
| --- | --- | --- |
| `03_durable_agent_run.py` | **Direct run** | Fire a workflow immediately from the CLI and print the final response. |
| `03_durable_agent_subscribe.py` | **Pub/Sub listener** | Keep the agent running so it reacts to pub/sub events (e.g., `@message_router`). |
| `03_durable_agent_serve.py` | **Service mode** | Host the agent as an HTTP service (still wired to pub/sub) and trigger workflows via REST. |

Each script defines the same TravelBuddy agent setup; use the command that matches the mode you need (remember to resolve the component templates first):

#### Direct Run
```bash
source .venv/bin/activate
dapr run --app-id stateful-llm --resources-path $temp_resources_folder -- python 03_durable_agent_run.py
```

#### Pub/Sub Listener
```bash
source .venv/bin/activate
dapr run --app-id stateful-llm --resources-path $temp_resources_folder -- python 03_durable_agent_subscribe.py
```

With the listener running, publish tasks using the included `message_client.py` (defaults to the `travel.requests` topic on the `messagepubsub` component):

```bash
source .venv/bin/activate
dapr run \
  --app-id travelbuddy-client \
  --resources-path $temp_resources_folder \
  -- python message_client.py "Plan a trip to Tokyo with two flight options"
```

#### Service Mode (HTTP + Pub/Sub)
```bash
dapr run \
  --app-id stateful-llm \
  --dapr-http-port 3500 \
  --app-port 8001 \
  --resources-path $temp_resources_folder \
  -- python 03_durable_agent_serve.py
```

When running `03_durable_agent_serve.py`, the runner installs default REST endpoints so you can trigger workflows without additional code:

**Start a new workflow:**

```bash
curl -i -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"task": "I want to find flights to Paris"}'
```

You'll receive a workflow ID in response, which you can use to track progress.

**Check workflow status:**

```bash
curl -i -X GET http://localhost:8001/run/WORKFLOW_ID

# You can also use the Dapr workflow API if you prefer:
curl -i -X GET http://localhost:3500/v1.0/workflows/dapr/WORKFLOW_ID
```

All three modes demonstrate the same durable travel-planning agent that:
1. Remembers user context persistently (across restarts)
2. Uses a tool to search for flight options
3. Reacts to pub/sub triggers (`messagepubsub` / `travel.requests`)
4. Exposes REST APIs for workflow interaction (service mode)
5. Stores execution state in Dapr workflow state stores

### How It Works

The key components of this implementation are:

1. **Persistent Memory**: The agent stores conversation state in Dapr's state store, enabling it to remember context across sessions and system restarts.

2. **Workflow Orchestration**: Long-running tasks are managed through Dapr's workflow system, providing:
    - Durability - workflows survive process crashes
    - Observability - track status and progress
    - Recoverability - automatic retry on failures

3. **Tool Integration**: A flight search tool is defined using the `@tool` decorator, which automatically handles input validation and type conversion.

4. **Trigger Exposure**: Depending on the entry point, the agent listens to pub/sub (`03_durable_agent_subscribe.py`), exposes REST endpoints (`03_durable_agent_serve.py`), or runs a single workflow directly (`03_durable_agent_run.py`).

### 4. Simple Workflow

Run the workflow example to see how to create a multi-step LLM process.

```bash
source .venv/bin/activate

dapr run --app-id dapr-agent-wf --resources-path $temp_resources_folder -- python 04_chain_tasks.py
```

This example demonstrates how to create a workflow with multiple tasks.

**Expected output:** The workflow will create an outline about AI Agents and then generate a blog post based on that outline.

### 5. Agent with Vector Store

**Prerequisites:** This example requires vectorstore dependencies. Install them using one of these methods:

**Using pip (recommended):**
```bash
pip install sentence-transformers chromadb 'posthog<6.0.0'
```

**Or install with extras:**
From the root directory,
```bash
pip install -e ".[vectorstore]"
```

**Using uv:**
```bash
uv add sentence-transformers chromadb 'posthog<6.0.0'
```

Run the vector store agent example to see how to create an agent that can search and store documents:
```bash
source .venv/bin/activate

python 05_agent_with_vectorstore.py
```

This example demonstrates how to create an agent with vector store capabilities, including logging, structured Document usage, and a tool to add a machine learning basics document.

## Key Concepts

- **DaprChatClient**: The interface for interacting with Dapr's LLMs
- **OpenAIChatClient**: The interface for interacting with OpenAI's LLMs
- **Agent**: A class that combines an LLM with tools and instructions
- **@tool decorator**: A way to create tools that agents can use
- **DurableAgent**: An agent that follows the Reasoning + Action pattern and achieves durability through Dapr Workflows
- **VectorStore**: Persistent storage for document embeddings that enables semantic search capabilities

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
