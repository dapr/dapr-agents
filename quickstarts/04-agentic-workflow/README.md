# Agentic Workflow with Task Chain

This quickstart demonstrates how to create stateful task chains using both pure Dapr Workflows and Dapr Agents' enhanced workflow capabilities. You'll learn how to orchestrate multiple tasks that use LLM inference, seeing firsthand how Dapr Agents simplifies and improves the workflow development experience.

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

1. Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

2. Replace `your_api_key_here` with your actual OpenAI API key.

3. Make sure Dapr is initialized on your system:

```bash
dapr init
```

4. Create the workflow state store component:

```bash
mkdir -p components
```

Add a `workflowstate.yaml` file to the components directory:

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

### 1. Dapr Workflow Without Dapr Agents

This example shows a basic task chain using pure Dapr workflows:

```python
# workflow_dapr.py
import dapr.ext.workflow as wf
from dotenv import load_dotenv
from openai import OpenAI
from time import sleep

# Load environment variables
load_dotenv()

# Initialize Workflow Instance
wfr = wf.WorkflowRuntime()

# Define Workflow logic
@wfr.workflow(name='lotr_workflow')
def task_chain_workflow(ctx: wf.DaprWorkflowContext):
    result1 = yield ctx.call_activity(get_character)
    result2 = yield ctx.call_activity(get_line, input=result1)
    return result2

# Activity 1
@wfr.activity(name='step1')
def get_character(ctx):
    client = OpenAI()
    response = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": "Pick a random character from The Lord of the Rings and respond with the character name only"
            }
        ],
        model = 'gpt-4o'
    )
    character = response.choices[0].message.content
    print(f"Character: {character}")
    return character

# Activity 2
@wfr.activity(name='step2')
def get_line(ctx, character: str):
    client = OpenAI()
    response = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": f"What is a famous line by {character}"
            }
        ],
        model = 'gpt-4o'
    )
    line = response.choices[0].message.content
    print(f"Line: {line}")
    return line

if __name__ == '__main__':
    wfr.start()
    sleep(5)  # wait for workflow runtime to start

    wf_client = wf.DaprWorkflowClient()
    instance_id = wf_client.schedule_new_workflow(workflow=task_chain_workflow)
    print(f'Workflow started. Instance ID: {instance_id}')
    state = wf_client.wait_for_workflow_completion(instance_id)
    print(f'Workflow completed! Status: {state.runtime_status}')

    wfr.shutdown()
```

Run the pure Dapr workflow:

<!-- STEP
name: Run text completion example
expected_stdout_lines:
  - "== APP == Workflow started. Instance ID:"
  - "== APP == Character:"
  - "== APP == Line:"
  - "== APP == Workflow completed! Status: WorkflowStatus.COMPLETED"
timeout_seconds: 30
output_match_mode: substring
-->
```bash
dapr run --app-id dapr-wf --resources-path components/ -- python workflow_dapr.py
```
<!-- END_STEP -->

**Expected output:** The workflow will select a random Lord of the Rings character and then generate a famous line by that character.

### 2. Dapr Agents Workflow

This example demonstrates how Dapr Agents simplifies the same workflow:

```python
# workflow_dapr_agent.py
from dapr_agents.workflow import WorkflowApp, workflow, task
from dapr_agents.types import DaprWorkflowContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the WorkflowApp

# Define Workflow logic
@workflow(name='task_chain_workflow')
def task_chain_workflow(ctx: DaprWorkflowContext):
    result1 = yield ctx.call_activity(get_character)
    result2 = yield ctx.call_activity(get_line, input={"character": result1})
    return result2

@task(description="""
    Pick a random character from The Lord of the Rings\n
    and respond with the character's name only
""")
def get_character() -> str:
    pass

@task(description="What is a famous line by {character}",)
def get_line(character: str) -> str:
    pass

if __name__ == '__main__':
    wfapp = WorkflowApp()

    results = wfapp.run_and_monitor_workflow(task_chain_workflow)
    print(f"Famous Line: {results}")
```

Run the Dapr Agents workflow:

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
dapr run --app-id dapr-agent-wf --resources-path components/ -- python workflow_dapr_agent.py
```
<!-- END_STEP -->

**Expected output:** Similar to the pure Dapr workflow, but with significantly less code.

## Key Concepts

- **Dapr Workflow**: A durable, resilient orchestration of activities
- **Workflow Activities**: Individual tasks within a workflow
- **Task Chain**: A sequence of tasks where each depends on the previous
- **WorkflowApp**: Dapr Agents' simplified workflow interface
- **LLM Tasks**: Automatically implemented tasks powered by LLMs

## Key Differences and Benefits

### Pure Dapr Workflow Approach
- Requires manual OpenAI client setup
- Explicit handling of API calls and responses
- More boilerplate code
- Direct control over workflow execution

### Dapr Agents Approach
- Automatic LLM client management
- Simplified task definitions using decorators
- Built-in prompt templating
- Reduced boilerplate code

## Dapr Integration

This quickstart demonstrates core Dapr capabilities:

- **Durability**: Workflows can survive process restarts
- **Actor Model**: Tasks run as reliable, stateful actors
- **Observability**: Workflow status tracking

## Troubleshooting

1. **Redis Connection**: Ensure Redis is running (automatically installed by Dapr)
2. **Dapr Initialization**: If components aren't found, verify Dapr is initialized with `dapr init`
3. **API Key**: Check your OpenAI API key if authentication fails

## Next Steps

After completing this quickstart, move on to the [Multi-Agent Workflow quickstart](../05-multi-agent-workflow/README.md) to learn how to create distributed systems of collaborating agents.