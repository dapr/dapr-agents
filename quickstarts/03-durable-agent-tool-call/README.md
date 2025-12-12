# Durable Agent Tool Call with Dapr Agents

This quickstart builds on the standalone version and shows how to run the same weather assistant as a **DurableAgent**. The agent logic stays the same, but execution happens inside Dapr Workflows—so runs are fault-tolerant, replayable, and can be triggered via pub/sub or HTTP.

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
dapr run --app-id durableweatherapp --resources-path $temp_resources_folder -- python durable_weather_agent_dapr.py

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

### Additional Components

The quickstart includes other necessary Dapr components in the `components` directory:

- `statestore.yaml`: Agent state configuration
- `pubsub.yaml`: Pub/Sub message bus configuration
- `workflowstate.yaml`: Workflow state configuration

Make sure Dapr is initialized on your system:

```bash
dapr init
```

## Running the Example
 
Durable agents maintain state across runs, enabling workflows that require persistence, recovery, and coordination. This is useful for long-running tasks, multi-step workflows, and agent collaboration.

Choose one of the following entry points depending on how you want to host the agent:

| Script | Mode | When to use it |
| --- | --- | --- |
| `durable_weather_agent_dapr.py` | **Direct run** | Fire a workflow immediately from the CLI and print the final response. |
| `durable_weather_agent_subscribe.py` | **Pub/Sub listener** | Keep the agent running so it reacts to pub/sub events (e.g., `@message_router`). |
| `durable_weather_agent_serve.py` | **Service mode** | Host the agent as an HTTP service (still wired to pub/sub) and trigger workflows via REST. |

Each script defines the same Weather Assistant agent setup; use the command that matches the mode you need (remember to resolve the component templates first):

### Direct Run
```bash
source .venv/bin/activate
dapr run --app-id durableweatherapp --resources-path $temp_resources_folder -- python durable_weather_agent_dapr.py
```

### Pub/Sub Listener
```bash
source .venv/bin/activate
dapr run --app-id durableweatherapp --resources-path $temp_resources_folder -- python durable_weather_agent_subscribe.py
```

With the listener running, publish tasks using the included `message_client.py` (defaults to the `weather.requests` topic on the `messagepubsub` component):

```bash
source .venv/bin/activate
dapr run \
  --app-id weather-client \
  --resources-path $temp_resources_folder \
  -- python message_client.py "What's the weather in Boston?"
```

### Service Mode (HTTP + Pub/Sub)
```bash
dapr run \
  --app-id durableweatherapp \
  --dapr-http-port 3500 \
  --app-port 8001 \
  --resources-path $temp_resources_folder \
  -- python durable_weather_agent_serve.py
```

When running `durable_weather_agent_serve.py`, the runner installs default REST endpoints so you can trigger workflows without additional code:

**Start a new workflow:**

```bash
curl -i -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What'\''s the weather in New York?"}'
```

You'll receive a workflow ID in response, which you can use to track progress.

**Check workflow status:**

```bash
curl -i -X GET http://localhost:8001/run/WORKFLOW_ID

# You can also use the Dapr workflow API if you prefer:
curl -i -X GET http://localhost:3500/v1.0/workflows/dapr/WORKFLOW_ID
```

## Other Durable Agent
You can also try the following Durable agents with the same tools using `OpenAI`, `HuggingFace hub` and `NVIDIA` LLM chat clients. Make sure you add the `OPENAI_API_KEY` and `HUGGINGFACE_API_KEY` and `NVIDIA_API_KEY` to the `.env` file.
- [OpenAI Durable Agent](./durable_weather_agent_openai.py)
- [HuggingFace Durable Agent](./durable_weather_agent_hf.py)
- [NVIDIA Durable Agent](./durable_weather_agent_nv.py)


All three modes demonstrate the same durable weather assistant agent that:
1. Remembers user context persistently (across restarts)
2. Uses tools to fetch weather and location information
3. Reacts to pub/sub triggers (`messagepubsub` / `weather.requests`)
4. Exposes REST APIs for workflow interaction (service mode)
5. Stores execution state in Dapr workflow state stores

### How It Works

The key components of this implementation are:

1. **Persistent Memory**: The agent stores conversation state in Dapr's state store, enabling it to remember context across sessions and system restarts.

2. **Workflow Orchestration**: Long-running tasks are managed through Dapr's workflow system, providing:
   - Durability - workflows survive process crashes
   - Observability - track status and progress
   - Recoverability - automatic retry on failures

3. **Tool Integration**: Weather and utility tools are defined using the `@tool` decorator, which automatically handles input validation and type conversion.

4. **Trigger Exposure**: Depending on the entry point, the agent listens to pub/sub (`durable_weather_agent_subscribe.py`), exposes REST endpoints (`durable_weather_agent_serve.py`), or runs a single workflow directly (`durable_weather_agent_dapr.py`).


## Custom Tools Example

See `agent_tools.py` for sample tool definitions.

## Observability with Phoenix Arize

This section demonstrates how to add observability to your Dapr Agents using Phoenix Arize for distributed tracing and monitoring. You'll learn how to set up Phoenix with PostgreSQL backend and instrument your agent for comprehensive observability.

### Phoenix Server Setup

First, deploy Phoenix Arize server using Docker Compose with PostgreSQL backend for persistent storage.

#### Prerequisites

- Docker and Docker Compose installed on your system
- Verify Docker is running: `docker info`

#### Deploy Phoenix with PostgreSQL

1. Use the provided [docker-compose.yml](./docker-compose.yml) file to set up Phoenix locally (PostgreSQL 18 + Phoenix).
2. Start the Phoenix server (this also provisions the required Postgres volume):

```bash
docker compose down -v   # optional: clean up old PG volumes
docker compose up --build
```

3. Verify Phoenix is running by navigating to [http://localhost:6006](http://localhost:6006)

#### Note on Production Deployment
For production deployments, ensure you:
- Use persistent volumes for PostgreSQL data
- Configure proper authentication and security
- Pin Phoenix version (e.g., `arizephoenix/phoenix:4.0.0`)

### Durable Agent Observability Setup

#### Install Observability Dependencies

Install the updated requirements:

```bash
pip install -r requirements.txt
```

#### Instrumented Weather Agent

See [`durable_weather_agent_tracing.py`](./durable_weather_agent_tracing.py) with Phoenix OpenTelemetry integration.

#### Run with Observability

1. Ensure Phoenix server is running (see setup steps above)

2. Run the instrumented Durable Agent:

```bash
dapr run --app-id durableweatherapptracing --resources-path $temp_resources_folder -- python durable_weather_agent_tracing.py
```

3. View traces in Phoenix UI at [http://localhost:6006](http://localhost:6006)

![durable agent span results](DurableAgentSpanResults.png)

![durable agent trace chat](DurableAgentChatMessage.png)

### Observability Features

Dapr Agents observability provides:

- **W3C Trace Context**: Standards-compliant distributed tracing
- **OpenTelemetry Integration**: Industry-standard instrumentation
- **Phoenix UI Compatibility**: Rich visualization and analysis
- **Automatic Instrumentation**: Zero-code tracing for agents and tools
- **Performance Monitoring**: Detailed metrics and performance insights
- **Error Tracking**: Comprehensive error capture and analysis

### Troubleshooting Observability

1. **Phoenix Connection Issues**: 
   - Verify Phoenix server is running: `docker compose ps`
   - Check port availability: `netstat -an | grep 6006`

2. **Missing Traces**:
   - Ensure `dapr-agents[observability]` is installed
   - Verify instrumentation is called before agent execution

3. **Docker Issues**:
   - Check Docker daemon is running: `docker info`
   - Verify PostgreSQL connectivity: `docker compose logs db`
