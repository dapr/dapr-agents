# Message Router Workflow (Pub/Sub → Workflow)

This quickstart shows how to trigger a Dapr Workflow from a Pub/Sub message using a lightweight `@message_router` decorator. Messages are validated at the edge (Pydantic), then your handler schedules a native Dapr workflow. Activities use the `@llm_activity` decorator to offload work to an LLM.

You’ll run two processes:

* **App**: subscribes to a topic, routes messages, and runs the workflow runtime
* **Client**: publishes a test message to that topic

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

## Project layout

```text
04-message-router-workflow/
├─ components/                 # Dapr components (pubsub, conversation, workflow state)
├─ app.py                      # Starts WorkflowRuntime + subscribes message routers
├─ handlers.py                 # @message_router handler (validates + schedules workflow)
├─ workflow.py                 # workflow & activities (decorated with @llm_activity)
└─ message_client.py           # publishes a test message to the topic
```

## Files

### workflow.py

```python
from __future__ import annotations

from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv

from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.decorators import llm_activity

load_dotenv()

# Initialize the LLM client and workflow runtime
llm = DaprChatClient(component_name="openai")


def blog_workflow(ctx: DaprWorkflowContext, wf_input: dict) -> str:
    """
    Workflow input must be JSON-serializable. We accept a dict like:
      {"topic": "<string>"}
    """
    topic = wf_input["topic"]
    outline = yield ctx.call_activity(create_outline, input={"topic": topic})
    post = yield ctx.call_activity(write_post, input={"outline": outline})
    return post


@llm_activity(
    prompt="Create a short outline about {topic}. Output 3-5 bullet points.",
    llm=llm,
)
async def create_outline(ctx, topic: str) -> str:
    # Implemented by the decorator; body can be empty.
    pass


@llm_activity(
    prompt="Write a short blog post following this outline:\n{outline}",
    llm=llm,
)
async def write_post(ctx, outline: str) -> str:
    # Implemented by the decorator; body can be empty.
    pass
```

### handlers.py

```python
from __future__ import annotations

import logging

import dapr.ext.workflow as wf
from dapr.clients.grpc._response import TopicEventResponse
from pydantic import BaseModel, Field

from dapr_agents.workflow.decorators.routers import message_router

logger = logging.getLogger(__name__)


class StartBlogMessage(BaseModel):
    topic: str = Field(min_length=1, description="Blog topic/title")


# Import the workflow after defining models to avoid circular import surprises
from workflow import blog_workflow  # noqa: E402


@message_router(pubsub="messagepubsub", topic="blog.requests")
def start_blog_workflow(message: StartBlogMessage) -> TopicEventResponse:
    """
    Triggered by pub/sub. Validates payload via Pydantic and schedules the workflow.
    """
    try:
        client = wf.DaprWorkflowClient()
        instance_id = client.schedule_new_workflow(
            workflow=blog_workflow,
            input=message.model_dump(),
        )
        logger.info("Scheduled blog_workflow instance=%s topic=%s", instance_id, message.topic)
        return TopicEventResponse("success")
    except Exception as exc:  # transient infra error → retry
        logger.exception("Failed to schedule blog workflow: %s", exc)
        return TopicEventResponse("retry")
```

### app.py

```python
from __future__ import annotations

import asyncio
import logging
import signal

import dapr.ext.workflow as wf
from dapr.clients import DaprClient
from dotenv import load_dotenv
from handlers import start_blog_workflow
from workflow import (
    blog_workflow,
    create_outline,
    write_post,
)

from dapr_agents.workflow.utils.registration import register_message_handlers

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _wait_for_shutdown() -> None:
    """Block until Ctrl+C or SIGTERM."""
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _set_stop(*_: object) -> None:
        stop.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _set_stop)
        loop.add_signal_handler(signal.SIGTERM, _set_stop)
    except NotImplementedError:
        # Windows fallback
        signal.signal(signal.SIGINT, lambda *_: _set_stop())
        signal.signal(signal.SIGTERM, lambda *_: _set_stop())

    await stop.wait()


async def main() -> None:
    runtime = wf.WorkflowRuntime()
    
    runtime.register_workflow(blog_workflow)
    runtime.register_activity(create_outline)
    runtime.register_activity(write_post)

    runtime.start()

    try:
        with DaprClient() as client:
            # Wire streaming subscriptions for our router(s)
            closers = register_message_handlers(
                targets=[start_blog_workflow],
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


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
```

### message_client.py

```python
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
import sys
from typing import Any, Dict

from dapr.clients import DaprClient

# ---------------------------
# Config via environment vars
# ---------------------------
PUBSUB_NAME = os.getenv("PUBSUB_NAME", "messagepubsub")
TOPIC_NAME = os.getenv("TOPIC_NAME", "blog.requests")
BLOG_TOPIC = os.getenv("BLOG_TOPIC", "AI Agents")  # used when RAW_DATA is not provided
RAW_DATA = os.getenv("RAW_DATA")  # if set, must be a JSON object (string)
CONTENT_TYPE = os.getenv("CONTENT_TYPE", "application/json")
CE_TYPE = os.getenv("CLOUDEVENT_TYPE")  # optional CloudEvent 'type' metadata

# Publish behavior
PUBLISH_ONCE = os.getenv("PUBLISH_ONCE", "true").lower() in {"1", "true", "yes"}
INTERVAL_SEC = float(os.getenv("INTERVAL_SEC", "0"))  # used when PUBLISH_ONCE=false
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "8"))
INITIAL_DELAY = float(os.getenv("INITIAL_DELAY", "0.5"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "2.0"))
JITTER_FRAC = float(os.getenv("JITTER_FRAC", "0.2"))

# Optional warmup (give sidecar/broker a moment)
STARTUP_DELAY = float(os.getenv("STARTUP_DELAY", "1.0"))

logger = logging.getLogger("publisher")


async def _backoff_sleep(delay: float, jitter: float, factor: float) -> float:
    """Sleep for ~delay seconds with ±jitter% randomness, then return the next delay."""
    actual = max(0.0, delay * (1 + random.uniform(-jitter, jitter)))
    if actual:
        await asyncio.sleep(actual)
    return delay * factor


def _build_payload() -> Dict[str, Any]:
    """
    Build the JSON payload:
      - if RAW_DATA is set → parse as JSON (must be an object)
      - else               → {"topic": BLOG_TOPIC}
    """
    if RAW_DATA:
        try:
            data = json.loads(RAW_DATA)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid RAW_DATA JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("RAW_DATA must be a JSON object")
        return data

    return {"topic": BLOG_TOPIC}


def _encode_payload(payload: Dict[str, Any]) -> bytes:
    """Encode the payload as UTF-8 JSON bytes."""
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


async def publish_once(client: DaprClient, payload: Dict[str, Any]) -> None:
    """Publish once with retries and exponential backoff."""
    delay = INITIAL_DELAY
    body = _encode_payload(payload)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info("publish attempt %d → %s/%s", attempt, PUBSUB_NAME, TOPIC_NAME)
            client.publish_event(
                pubsub_name=PUBSUB_NAME,
                topic_name=TOPIC_NAME,
                data=body,
                data_content_type=CONTENT_TYPE,
                publish_metadata=({"cloudevent.type": CE_TYPE} if CE_TYPE else None),
            )
            logger.info("published successfully")
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("publish failed: %s", exc)
            if attempt == MAX_ATTEMPTS:
                raise
            logger.info("retrying in ~%.2fs …", delay)
            delay = await _backoff_sleep(delay, JITTER_FRAC, BACKOFF_FACTOR)


async def main() -> int:
    logging.basicConfig(level=logging.INFO)
    stop_event = asyncio.Event()

    # Signal-aware shutdown
    loop = asyncio.get_running_loop()

    def _stop(*_: object) -> None:
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _stop)
        loop.add_signal_handler(signal.SIGTERM, _stop)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda *_: _stop())
        signal.signal(signal.SIGTERM, lambda *_: _stop())

    # Optional warmup
    if STARTUP_DELAY > 0:
        await asyncio.sleep(STARTUP_DELAY)

    payload = _build_payload()
    logger.info("payload: %s", payload)

    try:
        with DaprClient() as client:
            if PUBLISH_ONCE:
                await publish_once(client, payload)
                # brief wait so logs flush nicely under dapr
                await asyncio.sleep(0.2)
                return 0

            # periodic mode
            if INTERVAL_SEC <= 0:
                logger.error("INTERVAL_SEC must be > 0 when PUBLISH_ONCE=false")
                return 2

            logger.info("starting periodic publisher every %.2fs", INTERVAL_SEC)
            while not stop_event.is_set():
                try:
                    await publish_once(client, payload)
                except Exception as exc:  # noqa: BLE001
                    logger.error("giving up after %d attempts: %s", MAX_ATTEMPTS, exc)

                # wait for next tick or shutdown
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=INTERVAL_SEC)
                except asyncio.TimeoutError:
                    pass

            logger.info("shutdown requested; exiting")
            return 0

    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

## How it works (flow)

* `message_client.py` publishes a CloudEvent-style JSON payload to `topic=blog.requests` on pubsub=messagepubsub.
* `app.py` starts the Dapr Workflow runtime, registers `blog_workflow` + `activitie`s, and subscribes your `@message_router` handler.
* `handlers.py` receives and validates the message (`StartBlogMessage` via Pydantic), then calls `DaprWorkflowClient().schedule_new_workflow(...)`.
* `workflow.py` runs `blog_workflow`, calling two LLM-backed activities (`create_outline`, `write_post`) via `@llm_activity`.

## Running

Start the app (subscriber + workflow runtime)

```bash
rendered_components=$(../resolve_env_templates.py ./components)
dapr run \
  --app-id message-workflow \
  --resources-path "$rendered_components" \
  -- python app.py
rm -rf "$rendered_components"
```

Publish a test message (publisher)

```bash
rendered_components=$(../resolve_env_templates.py ./components)
dapr run \
  --app-id message-workflow-client \
  --resources-path "$rendered_components" \
  -- python message_client.py
rm -rf "$rendered_components"
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

### Additional Examples

Publish a custom topic via env:

```bash
BLOG_TOPIC="Serverless Agents" dapr run \
  --app-id message-workflow-client \
  --resources-path "$rendered_components" \
  -- python message_client.py
```

Publish a raw JSON object:

```bash
RAW_DATA='{"topic":"Thoughtful Orchestration"}' dapr run \
  --app-id message-workflow-client \
  --resources-path "$rendered_components" \
  -- python message_client.py
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
