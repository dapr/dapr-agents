# Agent-based Workflow Patterns

This quickstart demonstrates how to orchestrate agentic tasks using Dapr Workflows and the `@agent_activity` decorator from Dapr Agents. You’ll learn how to compose multi-step workflows that call autonomous agents—each powered by LLMs—for reasoning, decision-making, and task execution.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key
- Dapr CLI and Docker installed

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
dapr run --app-id dapr-agent-wf --resources-path $temp_resources_folder -- python 01_sequential_workflow.py

# Clean up when done
rm -rf $temp_resources_folder
```

> The temporary resources folder will be automatically deleted when the Dapr sidecar is stopped or when the computer is restarted.

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

> Many LLM providers are compatible with OpenAI's API (DeepSeek, Google AI, etc.) and can be used with this component by configuring the appropriate parameters. Dapr also has [native support](https://docs.dapr.io/reference/components-reference/supported-conversation/) for other providers like Google AI, Anthropic, Mistral, DeepSeek, etc.

### Additional Components

Make sure Dapr is initialized on your system:

```bash
dapr init
```

The quickstart includes other necessary Dapr components in the `components` directory. For example, the workflow state store component:

Look at the `workflowstate.yaml` file in the `components` directory:

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: workflowstatestore
spec:
  type: state.redis
  version: v1
  metadata:
  - name: redisHost
    value: localhost:6379
  - name: redisPassword
    value: ""
  - name: actorStateStore
    value: "true"
```

## Examples

### 1. Sequential Agent Chain (01_sequential_workflow.py)

This example chains three autonomous agents in a Dapr Workflow.
Each agent performs one stage of the reasoning process: extraction, planning, and expansion.

Workflow Overview

| Step | Agent | Responsibility |
| --- | --- | --- |
| 1️⃣ | DestinationExtractor | Identify the destination city from the user message |
| 2️⃣ | PlannerAgent | Create a concise 3-day outline for the destination |
| 3️⃣ | ItineraryAgent | Expand the outline into a detailed itinerary |

Run

```bash
dapr run --app-id dapr-agent-planner --resources-path components/ -- python 01_sequential_workflow.py
```

How It Works

* The workflow begins with the user message, e.g., `"Plan a trip to Paris."`
* `extract_destination` calls the `DestinationExtractor` agent to return the city name.
* `plan_outline` uses the `PlannerAgent` to generate a 3-day itinerary outline.
* `expand_itinerary` passes that outline to the `ItineraryAgent`, which expands it into a detailed plan.
* The final output is logged as a structured itinerary text.

#### Code Highlights

* `@agent_activity` decorator: Wraps an activity function so that Dapr automatically delegates its implementation to an Agent.
The function body can remain empty (pass); execution is routed through the agent’s reasoning loop.
* Agents: Each agent defines:
    * name, role, and instructions
    * a shared llm client (DaprChatClient)
    * internal memory, message history, and optional tools
* Workflow Orchestration: The `@runtime.workflow` function coordinates agent tasks:

```python
dest = yield ctx.call_activity(extract_destination, input=user_msg)
outline = yield ctx.call_activity(plan_outline, input=dest["content"])
itinerary = yield ctx.call_activity(expand_itinerary, input=outline["content"])
return itinerary["content"]
```

## Integration with Dapr

Dapr Agents workflows leverage Dapr's core capabilities:

- **Durability**: Workflows survive process restarts or crashes
- **State Management**: Workflow state is persisted in a distributed state store
- **Actor Model**: Tasks run as reliable, stateful actors within the workflow
- **Event Handling**: Workflows can react to external events

## Troubleshooting

1. **Docker is Running**: Ensure Docker is running with `docker ps` and verify you have container instances with `daprio/dapr`, `openzipkin/zipkin`, and `redis` images running
2. **Redis Connection**: Ensure Redis is running (automatically installed by Dapr)
3. **Dapr Initialization**: If components aren't found, verify Dapr is initialized with `dapr init`
4. **API Key**: Check your OpenAI API key if authentication fails
5. **GRPC Deadline Exceeded**: Set your `DAPR_API_TIMEOUT_SECONDS` environment variable to `300` so the Dapr gRPC client waits longer than the default 60 s before timing out long LLM calls.

## Next Steps

After completing this quickstart, move on to the [Multi-Agent Workflow quickstart](../05-multi-agent-workflows/README.md) to learn how to create distributed systems of collaborating agents.