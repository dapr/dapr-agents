
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

### 1. Sequential Task Execution

This example demonstrates the Chaining Pattern by executing two activities in sequence.

Run the sequential task chain workflow:
```bash
dapr run --app-id dapr-agent-wf-sequence --resources-path components/ -- python 03_sequential_workflow.py
```

**How it works:**
In this chaining pattern, the workflow executes tasks in strict sequence:
1. The `get_character()` task executes first and returns a character name
2. Only after completion, the `get_line()` task runs using that character name as input
3. Each task awaits the previous task's completion before starting

### 2. Parallel Task Execution

This example demonstrates the Fan-out/Fan-in Pattern with a research use case. It will execute 3 activities in parallel; then synchronize these activities do not proceed with the execution of subsequent activities until all preceding activities have completed.

Run the parallel research workflow:
```bash
dapr run --app-id dapr-agent-research --resources-path components/ -- python 04_parallel_workflow.py
```

**How it works:**
This fan-out/fan-in pattern combines sequential and parallel execution:
1. First, `generate_questions()` executes sequentially
2. Multiple `gather_information()` tasks run in parallel using `wf.when_all()`
3. The workflow waits for all parallel tasks to complete
4. Finally, `synthesize_results()` executes with all gathered data

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

## Next Steps

After completing this quickstart, move on to the [Agent Based Workflow Quickstart](../04-agent-absed-workflows/README.md) to learn how to integrate the concept of an agent on specific activity steps.