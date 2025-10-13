
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

Each script shows a distinct workflow pattern using the decorators
(`@runtime.workflow`, `@runtime.activity`, and `@llm_activity`).

### 1. Single LLM Activity (01_single_activity_workflow.py)

A simple example where the workflow executes one LLM-powered activity that returns a short biography.

```python
import time

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv

from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.decorators import llm_activity

# Load environment variables (e.g., API keys, secrets)
load_dotenv()

# Initialize the Dapr workflow runtime and LLM client
runtime = wf.WorkflowRuntime()
llm = DaprChatClient(component_name="openai")


@runtime.workflow(name="single_task_workflow")
def single_task_workflow(ctx: DaprWorkflowContext, name: str):
    """Ask the LLM about a single historical figure and return a short bio."""
    response = yield ctx.call_activity(describe_person, input={"name": name})
    return response


@runtime.activity(name="describe_person")
@llm_activity(
    prompt="Who was {name}?",
    llm=llm,
)
async def describe_person(ctx, name: str) -> str:
    pass


if __name__ == "__main__":
    runtime.start()
    time.sleep(5)

    client = wf.DaprWorkflowClient()
    instance_id = client.schedule_new_workflow(
        workflow=single_task_workflow,
        input="Grace Hopper",
    )
    print(f"Workflow started: {instance_id}")

    state = client.wait_for_workflow_completion(instance_id)
    if not state:
        print("No state returned (instance may not exist).")
    elif state.runtime_status.name == "COMPLETED":
        print(f"Grace Hopper bio:\n{state.serialized_output}")
    else:
        print(f"Workflow ended with status: {state.runtime_status}")
        if state.failure_details:
            fd = state.failure_details
            print("Failure type:", fd.error_type)
            print("Failure message:", fd.message)
            print("Stack trace:\n", fd.stack_trace)
        else:
            print("Custom status:", state.serialized_custom_status)

    runtime.shutdown()
```

Run

```bash
dapr run --app-id dapr-agent-wf --resources-path components/ -- python 01_single_activity_workflow.py
```

How it works:

* The single_task_workflow calls one LLM activity (describe_person).
* The `@llm_activity` decorator automatically invokes the LLM with `"Who was {name}?"`.
* The workflow returns the model's response (a short bio).

### 2. Structured LLM Output (02_single_structured_activity_workflow.py)

Demonstrates schema-guided prompting using Pydantic models.

```python
import time

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv
from pydantic import BaseModel

from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.decorators import llm_activity


class Dog(BaseModel):
    name: str
    bio: str
    breed: str


# Load environment variables (e.g., API keys, secrets)
load_dotenv()

# Initialize the Dapr workflow runtime and LLM client
runtime = wf.WorkflowRuntime()
llm = DaprChatClient(component_name="openai")


@runtime.workflow(name="single_task_workflow_structured")
def single_task_workflow_structured(ctx: DaprWorkflowContext, name: str):
    """Ask the LLM for structured data about a dog and return the result."""
    result = yield ctx.call_activity(describe_dog, input={"name": name})
    return result


@runtime.activity(name="describe_dog")
@llm_activity(
    prompt="""
You are a JSON-only API. Return a Dog object for the dog named {name}."
JSON schema (informal):
{{
    "name":  string,   // Dog's full name
    "bio":   string,   // 1-3 sentence biography
    "breed": string    // Primary breed or mixed
}}
""",
    llm=llm)
def describe_dog(ctx, name: str) -> Dog:
    pass


if __name__ == "__main__":
    runtime.start()
    time.sleep(5)

    client = wf.DaprWorkflowClient()
    instance_id = client.schedule_new_workflow(
        workflow=single_task_workflow_structured,
        input="Laika",
    )
    print(f"Workflow started: {instance_id}")

    state = client.wait_for_workflow_completion(instance_id)
    if not state:
        print("No state returned (instance may not exist).")
    elif state.runtime_status.name == "COMPLETED":
        print(f"Dog Bio:\n{state.serialized_output}")
    else:
        print(f"Workflow ended with status: {state.runtime_status}")
        if state.failure_details:
            fd = state.failure_details
            print("Failure type:", fd.error_type)
            print("Failure message:", fd.message)
            print("Stack trace:\n", fd.stack_trace)
        else:
            print("Custom status:", state.serialized_custom_status)

    runtime.shutdown()
```

Run

```bash
dapr run --app-id dapr-agent-wf-structured --resources-path components/ -- python 02_single_structured_activity_workflow.py
```

How it works:

* Defines a Pydantic model Dog for structured output.
* The `describe_dog` activity uses `@llm_activity` to return a JSON-like object mapped to Dog.
* The workflow yields structured, validated data (name, bio, breed).

### 3. Sequential Task Chain (03_sequential_workflow.py)

Chains multiple LLM-powered activities where the output of one becomes the input of the next.

```python
import time

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv

from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.decorators import llm_activity

# Load environment variables (e.g., API keys, secrets)
load_dotenv()

# Initialize the Dapr workflow runtime and LLM client
runtime = wf.WorkflowRuntime()
llm = DaprChatClient(component_name="openai")


@runtime.workflow(name="task_chain_workflow")
def task_chain_workflow(ctx: DaprWorkflowContext):
    """
    Chain two LLM-backed activities:
      1) Pick a random LOTR character (name only)
      2) Ask for a famous quote from that character
    """
    character = yield ctx.call_activity(get_character)
    line = yield ctx.call_activity(get_line, input={"character": character})
    return line


@runtime.activity(name="get_character")
@llm_activity(
    prompt="""
Pick a random character from The Lord of the Rings.
Respond with the character's name only.
""",
    llm=llm,
)
def get_character(ctx) -> str:
    # The llm_activity decorator handles the LLM call using the prompt above.
    # Just declare the signature; the body can be empty or 'pass'.
    pass


@runtime.activity(name="get_line")
@llm_activity(
    prompt="What is a famous line by {character}?",
    llm=llm,
)
def get_line(ctx, character: str) -> str:
    # The llm_activity decorator will format the prompt with 'character'.
    pass


if __name__ == "__main__":
    # Start the workflow runtime sidecar
    runtime.start()
    time.sleep(5)  # small grace period for runtime to be ready

    # Kick off the workflow
    client = wf.DaprWorkflowClient()
    instance_id = client.schedule_new_workflow(
        workflow=task_chain_workflow,
        input=None,  # no input expected for this workflow
    )
    print(f"Workflow started: {instance_id}")

    # Wait for completion and print results
    state = client.wait_for_workflow_completion(instance_id)
    if not state:
        print("No state returned (instance may not exist).")
    elif state.runtime_status.name == "COMPLETED":
        print(f"Famous Line:\n{state.serialized_output}")
    else:
        print(f"Workflow ended with status: {state.runtime_status}")
        if state.failure_details:
            fd = state.failure_details
            print("Failure type:", fd.error_type)
            print("Failure message:", fd.message)
            print("Stack trace:\n", fd.stack_trace)
        else:
            print("Custom status:", state.serialized_custom_status)

    runtime.shutdown()
```

Run the sequential task chain workflow:

<!-- STEP
name: Run text completion example
expected_stdout_lines:
  - "== APP == Character:"
  - "== APP == Line:"
  - "== APP == Results:"
timeout_seconds: 30
output_match_mode: substring
-->
```bash
dapr run --app-id dapr-agent-wf-sequence --resources-path components/ -- python 03_sequential_workflow.py
```
<!-- END_STEP -->

How it works:

* The workflow first calls `get_character()` — the LLM picks a random LOTR character.
* Once complete, it calls `get_line()` to fetch a famous quote from that character.
* The final result is returned to the caller.

This pattern ensures strict sequential execution, waiting for each activity to finish before continuing.

### 4. Parallel Fan-out/Fan-in Workflow (04_parallel_workflow.py)

Demonstrates how to run multiple LLM activities concurrently and synthesize their results.

```python
import logging
import time
from typing import List

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.decorators import llm_activity

# Load environment variables (API keys, etc.)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the Dapr workflow runtime and LLM client
runtime = wf.WorkflowRuntime()
llm = DaprChatClient(component_name="openai")


# ----- Models -----

class Question(BaseModel):
    """Represents a single research question."""
    text: str = Field(..., description="A research question related to the topic.")


class Questions(BaseModel):
    """Encapsulates a list of research questions."""
    questions: List[Question] = Field(
        ..., description="A list of research questions generated for the topic."
    )


# ----- Workflow -----

@runtime.workflow(name="research_workflow")
def research_workflow(ctx: DaprWorkflowContext, topic: str):
    """Defines a Dapr workflow for researching a given topic."""
    # 1) Generate research questions
    questions: Questions = yield ctx.call_activity(
        generate_questions, input={"topic": topic}
    )

    # Handle both dict and model cases gracefully
    q_list = (
        [q["text"] for q in questions["questions"]]
        if isinstance(questions, dict)
        else [q.text for q in questions.questions]
    )

    # 2) Gather information for each question in parallel
    parallel_tasks = [
        ctx.call_activity(gather_information, input={"question": q})
        for q in q_list
    ]
    research_results: List[str] = yield wf.when_all(parallel_tasks)

    # 3) Synthesize final report
    final_report: str = yield ctx.call_activity(
        synthesize_results, input={"topic": topic, "research_results": research_results}
    )

    return final_report


# ----- Activities -----

@runtime.activity(name="generate_questions")
@llm_activity(
    prompt="""
You are a research assistant. Generate exactly 3 focused research questions about the topic: {topic}.
Return ONLY a JSON object matching this schema (no prose):

{
  "questions": [
    { "text": "..." },
    { "text": "..." },
    { "text": "..." }
  ]
}
""",
    llm=llm,
)
def generate_questions(ctx, topic: str) -> Questions:
    # Implemented by llm_activity via the prompt above.
    pass


@runtime.activity(name="gather_information")
@llm_activity(
    prompt="""
Research the following question and provide a detailed, well-cited answer (paragraphs + bullet points where helpful).
Question: {question}
""",
    llm=llm,
)
def gather_information(ctx, question: str) -> str:
    # Implemented by llm_activity via the prompt above.
    pass


@runtime.activity(name="synthesize_results")
@llm_activity(
    prompt="""
Create a comprehensive research report on the topic "{topic}" using the following research findings:

{research_results}

Requirements:
- Clear executive summary (3–5 sentences)
- Key findings (bulleted)
- Risks/unknowns
- Short conclusion

Return plain text (no JSON).
""",
    llm=llm,
)
def synthesize_results(ctx, topic: str, research_results: List[str]) -> str:
    # Implemented by llm_activity via the prompt above.
    pass


# ----- Entrypoint -----

if __name__ == "__main__":
    runtime.start()
    time.sleep(5)  # small grace period for runtime readiness

    client = wf.DaprWorkflowClient()
    research_topic = "The environmental impact of quantum computing"

    logging.info(f"Starting research workflow on: {research_topic}")
    instance_id = client.schedule_new_workflow(
        workflow=research_workflow,
        input=research_topic,
    )
    logging.info(f"Workflow started: {instance_id}")

    state = client.wait_for_workflow_completion(instance_id)
    if not state:
        logging.error("No state returned (instance may not exist).")
    elif state.runtime_status.name == "COMPLETED":
        logging.info(f"\nResearch Report:\n{state.serialized_output}")
    else:
        logging.error(f"Workflow ended with status: {state.runtime_status}")
        if state.failure_details:
            fd = state.failure_details
            logging.error("Failure type: %s", fd.error_type)
            logging.error("Failure message: %s", fd.message)
            logging.error("Stack trace:\n%s", fd.stack_trace)
        else:
            logging.error("Custom status: %s", state.serialized_custom_status)

    runtime.shutdown()
```

Run the parallel research workflow:

<!-- STEP
name: Run parallel workflows example
expected_stdout_lines:
  - "Starting research workflow on: The environmental impact of quantum computing"
  - "Research Report:"
output_match_mode: substring
-->
```bash
dapr run --app-id dapr-agent-research --resources-path components/ -- python 04_parallel_workflow.py
```
<!-- END_STEP -->

**How it works:**
This fan-out/fan-in pattern combines sequential and parallel execution:
1. First, `generate_questions()` executes sequentially
2. Multiple `gather_information()` tasks run in parallel using `ctx.when_all()`
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