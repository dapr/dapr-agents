# Agents as Tasks in Workflows with Dapr Agents

This quickstart demonstrates how to use AI agents as tasks within Dapr workflows, enabling sophisticated orchestration of agent-based automation. You'll learn how to build resilient, stateful workflows that leverage agents for intelligent task execution, decision-making, and multi-step reasoning.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key (or NVIDIA API key and Hugging Face API key for multi-model examples)
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

The quickstart includes component configurations in the `components` directory for various LLM providers. You have two options to configure your API keys:

### Option 1: Using Environment Variables (Recommended)

1. Create a `.env` file in the project root and add your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here  # Optional, for multi-model examples
HUGGINGFACE_API_KEY=your_huggingface_api_key_here  # Optional, for multi-model examples
```

2. When running the examples with Dapr, use the helper script to resolve environment variables:
```bash
# Get the environment variables from the .env file:
export $(grep -v '^#' ../../.env | xargs)

# Create a temporary resources folder with resolved environment variables
temp_resources_folder=$(../resolve_env_templates.py ./components)

# Run your dapr command with the temporary resources
dapr run --app-id agent-workflow --resources-path $temp_resources_folder -- python sequential_workflow.py

# Clean up when done
rm -rf $temp_resources_folder
```

Note: The temporary resources folder will be automatically deleted when the Dapr sidecar is stopped or when the computer is restarted.

### Option 2: Direct Component Configuration

You can directly update the API keys in the respective component files:

1. For OpenAI ([components/openai.yaml](components/openai.yaml)):
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

2. For NVIDIA ([components/nvidia.yaml](components/nvidia.yaml)):
```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: nvidia
spec:
  type: conversation.nvidia
  metadata:
    - name: key
      value: "YOUR_NVIDIA_API_KEY"
```

3. For Hugging Face ([components/huggingface.yaml](components/huggingface.yaml)):
```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: huggingface
spec:
  type: conversation.huggingface
  metadata:
    - name: key
      value: "YOUR_HUGGINGFACE_API_KEY"
```

Replace `YOUR_*_API_KEY` with your actual API keys.

Note: Many LLM providers are compatible with OpenAI's API (DeepSeek, Google AI, etc.) and can be used with the OpenAI component by configuring the appropriate parameters. Dapr also has [native support](https://docs.dapr.io/reference/components-reference/supported-conversation/) for other providers like Google AI, Anthropic, Mistral, DeepSeek, etc.

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

> ⏱️ **Long-running prompts?** Set `DAPR_API_TIMEOUT_SECONDS=300` in your shell to let the Dapr gRPC client wait longer than the default 60 seconds when scheduling or monitoring workflows.

## Sequential Agent Workflow

This example shows how to chain multiple agents inside a Dapr workflow using the `@agent_activity` decorator. Each activity runs an agent with its own instructions, while the workflow orchestrates the overall plan.

Run the workflow (render components first if you’re using `.env` placeholders):

```bash
temp_resources_folder=$(../resolve_env_templates.py ./components)
dapr run \
  --app-id agent-workflow \
  --resources-path "$temp_resources_folder" \
  -- python sequential_workflow.py
rm -rf "$temp_resources_folder"
```

The quickstart also includes tracing variants (`sequential_workflow_tracing.py`) and a multi-model version (`sequential_workflow_multi_model_tracing.py`) that shows how to instrument Phoenix OpenTelemetry and mix OpenAI/NVIDIA/Hugging Face agents in a single workflow.

Run them the same way, swapping the script name:

```bash
dapr run --app-id agent-workflow --resources-path $temp_resources_folder -- python sequential_workflow_tracing.py
dapr run --app-id agent-workflow --resources-path $temp_resources_folder -- python sequential_workflow_multi_model_tracing.py
```

**How it works:**
1. The `extract` task uses an agent to extract the destination from the user query
2. The `plan` task uses a different agent to create a 3-day outline for that destination  
3. The `expand` task uses a third agent to expand the outline into a detailed itinerary
4. Each agent has specialized instructions and capabilities for its specific role

## Observability with Phoenix Arize

This section demonstrates how to add observability to your Dapr Agent workflows using Phoenix Arize for distributed tracing and monitoring. You'll learn how to set up Phoenix with PostgreSQL backend and instrument your workflow for comprehensive observability.

### Phoenix Server Setup

First, deploy Phoenix Arize server using Docker Compose with PostgreSQL backend for persistent storage.

#### Prerequisites

- Docker and Docker Compose installed on your system
- Verify Docker is running: `docker info`

#### Deploy Phoenix with PostgreSQL

1. Use the [docker-compose.yml](./docker-compose.yml) file provided to set up a Phoenix server locally with PostgreSQL backend.

2. Start the Phoenix server:

```bash
docker compose up --build
```

3. Verify Phoenix is running by navigating to [http://localhost:6006](http://localhost:6006)

#### Install Observability Dependencies

Install the updated requirements:

```bash
pip install -r requirements.txt
```

### Examples

#### 1. Sequential Agent Workflow with Tracing

This example adds observability to the sequential workflow using Phoenix Arize for distributed tracing.

Run with tracing:

```bash
dapr run --app-id agent-workflow-tracing --resources-path components/ -- python sequential_workflow_tracing.py
```

View traces in Phoenix UI at [http://localhost:6006](http://localhost:6006)


![Agents as Tasks Sequence](./agents-as-tasks-tracing.png)


#### 2. Multi-Model Sequential Workflow with Tracing

This example demonstrates using multiple LLM providers within a single workflow, with full observability.

Run with tracing:

```bash
dapr run --app-id multi-model-workflow --resources-path components/ -- python sequential_workflow_multi_model_tracing.py
```

View traces in Phoenix UI at [http://localhost:6006](http://localhost:6006)

![Agents as Tasks Sequence (Multi-Model) with OpenAI endpoint and Tracing](./agents-as-tasks-multi-model-tracing-oai.png)

![Agents as Tasks Sequence (Multi-Model) with NVIDIA endpoint and Tracing](./agents-as-tasks-multi-model-tracing-nv.png)

![Agents as Tasks Sequence (Multi-Model) with HuggingFace Hub endpoint and Tracing](./agents-as-tasks-multi-model-tracing-hf.png)

## Key Concepts

### Agent as Task Pattern
- Agents can be used directly as workflow tasks using the `@task(agent=agent_instance)` decorator
- Each agent operates independently with its own instructions and capabilities
- Agents receive input from previous workflow steps and pass output to subsequent steps

### Workflow Orchestration  
- The `@workflow` decorator defines the orchestration logic
- `yield ctx.call_activity()` executes tasks sequentially
- Agent results are automatically serialized and passed between tasks

### Multi-Model Integration
- Different agents can use different LLM providers within the same workflow
- Enables leveraging the strengths of different models for specific tasks
- All interactions are traced for comprehensive observability

### Observability Features

Dapr Agent workflows with observability provide:

- **W3C Trace Context**: Standards-compliant distributed tracing across workflow steps
- **OpenTelemetry Integration**: Industry-standard instrumentation for workflows and agents
- **Phoenix UI Compatibility**: Rich visualization of workflow execution and agent interactions
- **Automatic Instrumentation**: Zero-code tracing for workflow tasks and agent operations
- **Performance Monitoring**: Detailed metrics for workflow execution and agent performance
- **Error Tracking**: Comprehensive error capture across the entire workflow

## Integration with Dapr

Dapr Agent workflows leverage Dapr's core capabilities:

- **Durability**: Workflows survive process restarts or crashes
- **State Management**: Workflow state is persisted in a distributed state store  
- **Actor Model**: Agent tasks run as reliable, stateful actors within the workflow
- **Event Handling**: Workflows can react to external events and agent responses

## Troubleshooting

1. **Docker is Running**: Ensure Docker is running with `docker ps` and verify you have container instances with `daprio/dapr`, `openzipkin/zipkin`, and `redis` images running
2. **Redis Connection**: Ensure Redis is running (automatically installed by Dapr)
3. **Dapr Initialization**: If components aren't found, verify Dapr is initialized with `dapr init`
4. **API Keys**: Check your OpenAI/NVIDIA API keys if authentication fails
5. **Phoenix Connection Issues**: 
   - Verify Phoenix server is running: `docker compose ps`
   - Check port availability: `netstat -an | grep 6006`
6. **Missing Traces**:
   - Ensure `dapr-agents[observability]` is installed
   - Verify instrumentation is called before workflow execution
