
# LLM-based Workflow Patterns

This quickstart demonstrates how to orchestrate sequential and parallel tasks using Dapr Agents' workflow capabilities powered by Language Models (LLMs). You'll learn how to build resilient, stateful workflows that leverage LLMs for reasoning, structured output, and automation, all using the new `@llm_activity` decorator and native Dapr workflow runtime.

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
dapr run --app-id dapr-agent-wf --resources-path $temp_resources_folder -- python sequential_workflow.py

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

### 1. Single LLM Activity (Free-Form)

Run the simplest possible LLM workflow—one activity that asks for a short biography.

```bash
dapr run --app-id dapr-agent-wf-single --resources-path $temp_resources_folder -- python 01_single_activity_workflow.py
```

**Why start here?**
- Shows how to define a workflow + activity with `WorkflowRuntime`
- Demonstrates `@llm_activity` returning plain text
- Uses `DaprWorkflowClient` to schedule and await a single run

### 2. Single LLM Activity (Structured Output)

Extend the previous sample by enforcing a JSON schema with Pydantic.

```bash
dapr run --app-id dapr-agent-wf-structured --resources-path $temp_resources_folder -- python 02_single_structured_activity_workflow.py
```

**Key ideas**
- `@llm_activity` can deserialize into typed models (e.g., `Dog`)
- Perfect for downstream steps that expect strongly typed data

### 3. Sequential Task Execution

This workflow chains two LLM activities: pick a LOTR character, then fetch a famous quote.

```bash
dapr run --app-id dapr-agent-wf-sequence --resources-path $temp_resources_folder -- python 03_sequential_workflow.py
```

**How it works**
1. `get_character()` runs first and returns a random character
2. Once it completes, `get_line()` runs with the character as input
3. Each activity waits for the previous one to finish before starting

### 4. Parallel Task Execution

The fan-out/fan-in pattern for research: generate questions, gather answers in parallel, then synthesize a report.

```bash
dapr run --app-id dapr-agent-research --resources-path $temp_resources_folder -- python 04_parallel_workflow.py
```

**How it works**
1. `generate_questions()` runs sequentially to produce three prompts
2. Multiple `gather_information()` activities run concurrently via `wf.when_all()`
3. The workflow waits for all parallel calls to finish
4. `synthesize_results()` produces the final report

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
5. **gRPC Timeout**: For longer prompts/responses set `DAPR_API_TIMEOUT_SECONDS=300` so the Dapr client waits beyond the 60 s default.

## Next Steps

After completing this quickstart, move on to the [Agent-Based Workflow Quickstart](../04-agent-based-workflows/README.md) to learn how to integrate full agents into workflow steps.
