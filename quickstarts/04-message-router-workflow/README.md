# Message Router Workflow (Pub/Sub → Workflow)

This quickstart shows how to trigger a Dapr Workflow directly from a Pub/Sub message using the `@message_router` decorator. The decorator is applied to the workflow itself, enabling automatic message validation and workflow scheduling. Activities use the `@llm_activity` decorator to offload work to an LLM.

You'll run two processes:

* **App**: subscribes to a topic and runs the workflow runtime
* **Client**: publishes a test message to that topic

## Key Concept

The `@message_router` decorator is applied **directly to the workflow function**, not to a separate handler. This means:
- The workflow IS the pub/sub handler
- Messages are automatically validated against the specified model
- The workflow is automatically scheduled when messages arrive
- No manual workflow scheduling code needed

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
dapr run --app-id dapr-agent-wf --resources-path $temp_resources_folder -- python workflow.py

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

## Project layout

```text
04-message-router-workflow/
├─ components/                 # Dapr components (pubsub, conversation, workflow state)
├─ app.py                      # Starts WorkflowRuntime + registers message router
├─ workflow.py                 # @message_router decorated workflow & @llm_activity activities
└─ message_client.py           # publishes a test message to the topic
```

## How it works (flow)

* `message_client.py` publishes a CloudEvent-style JSON payload to `topic=blog.requests` on `pubsub=messagepubsub`.
* `app.py` starts the Dapr Workflow runtime, registers `blog_workflow` + activities, and calls `register_message_routes(targets=[blog_workflow])`.
* `register_message_routes` discovers the `@message_router` decorator on `blog_workflow`, validates incoming messages using the Pydantic model (`StartBlogMessage`), and automatically schedules the workflow when valid messages arrive.
* `workflow.py` runs `blog_workflow`, calling two LLM-backed activities (`create_outline`, `write_post`) decorated with `@llm_activity`.

## Code Structure

### app.py

The application entry point registers the workflow and sets up the pub/sub subscription:

```python
async def main() -> None:
    runtime = wf.WorkflowRuntime()
    
    # Register the workflow (which is also the message handler)
    runtime.register_workflow(blog_workflow)
    runtime.register_activity(create_outline)
    runtime.register_activity(write_post)

    runtime.start()

    try:
        with DaprClient() as client:
            # Register the pub/sub subscription
            # register_message_routes discovers the @message_router decorator
            # and automatically sets up subscription + validation
            closers = register_message_routes(
                targets=[blog_workflow],  # Pass the workflow itself
                dapr_client=client,
            )

            try:
                await _wait_for_shutdown()
            finally:
                for close in closers:
                    try:
                        close()
                    except Exception:
                        logger.exception("Error while closing subscription")
    finally:
        runtime.shutdown()
```

**Key Points:**
- `runtime.register_workflow(blog_workflow)` - Register the workflow with the runtime
- `register_message_routes(targets=[blog_workflow])` - Set up pub/sub subscription by discovering the `@message_router` decorator
- The workflow function itself is passed as the target, not a separate handler function
- When a message arrives, `register_message_routes` validates it and automatically schedules the workflow

## Running

Start the app (subscriber + workflow runtime)

```bash
dapr run \
  --app-id message-workflow \
  --resources-path $temp_resources_folder \
  -- python app.py
rm -rf $temp_resources_folder
```

Publish a test message (publisher)

```bash
dapr run \
  --app-id message-workflow-client \
  --resources-path $temp_resources_folder \
  -- python message_client.py
rm -rf $temp_resources_folder
```

## Publisher configuration (env vars)

You can tweak message_client.py using environment variables:

| Variable          | Default            | Description                                         |
| ----------------- | ------------------ | --------------------------------------------------- |
| `PUBSUB_NAME`     | `messagepubsub`    | Pub/Sub component name                              |
| `TOPIC_NAME`      | `blog.requests`    | Topic to publish to                                 |
| `BLOG_TOPIC`      | `AI Agents`        | Fallback payload: `{"topic": BLOG_TOPIC}`           |
| `RAW_DATA`        | *(unset)*          | JSON string that overrides payload (must be object) |
| `CONTENT_TYPE`    | `application/json` | Content type sent with the event                    |
| `CLOUDEVENT_TYPE` | *(unset)*          | Optional `cloudevent.type` metadata                 |
| `PUBLISH_ONCE`    | `true`             | If `false`, publish periodically                    |
| `INTERVAL_SEC`    | `0`                | Period (seconds) when `PUBLISH_ONCE=false`          |
| `MAX_ATTEMPTS`    | `8`                | Retry attempts per publish                          |
| `INITIAL_DELAY`   | `0.5`              | Initial backoff seconds                             |
| `BACKOFF_FACTOR`  | `2.0`              | Exponential backoff factor                          |
| `JITTER_FRAC`     | `0.2`              | ± jitter applied to each delay                      |
| `STARTUP_DELAY`   | `1.0`              | Sleep before first publish (sidecar warmup)         |

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
