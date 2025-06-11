# Agentic Workflow Patterns

This quickstart demonstrates how to orchestrate sequential and parallel tasks using Dapr Agents' workflow capabilities.

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

This example demonstrates the Chaining Pattern by executing two activities in sequence:

```python
from dapr_agents.workflow import WorkflowApp, workflow, task
from dapr.ext.workflow import DaprWorkflowContext
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

    results = wfapp.run_and_monitor_workflow_sync(task_chain_workflow)
    print(f"Famous Line: {results}")
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
dapr run --app-id dapr-agent-wf --resources-path components/ -- python sequential_workflow.py 
```
<!-- END_STEP -->

**How it works:**
In this chaining pattern, the workflow executes tasks in strict sequence:
1. The `get_character()` task executes first and returns a character name
2. Only after completion, the `get_line()` task runs using that character name as input
3. Each task awaits the previous task's completion before starting

### 2. Parallel Task Execution

This example demonstrates the Fan-out/Fan-in Pattern with a research use case. It will execute 3 activities in parallel; then synchronize these activities do not proceed with the execution of subsequent activities until all preceding activities have completed.

```python
import logging
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dapr_agents.workflow import WorkflowApp, workflow, task
from dapr.ext.workflow import DaprWorkflowContext

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)


# Define a structured model for a single question
class Question(BaseModel):
    """Represents a single research question."""
    text: str = Field(..., description="A research question related to the topic.")


# Define a model that holds multiple questions
class Questions(BaseModel):
    """Encapsulates a list of research questions."""
    questions: List[Question] = Field(...,
                                      description="A list of research questions generated for the topic.")


# Define Workflow logic
@workflow(name="research_workflow")
def research_workflow(ctx: DaprWorkflowContext, topic: str):
    """Defines a Dapr workflow for researching a given topic."""

    # Generate research questions
    questions: Questions = yield ctx.call_activity(generate_questions, input={"topic": topic})

    # Gather information for each question in parallel
    parallel_tasks = [ctx.call_activity(gather_information, input={"question": q["text"]}) for q in
        questions["questions"]]
    research_results = yield wfapp.when_all(parallel_tasks)  # Ensure wfapp is initialized

    # Synthesize the results into a final report
    final_report = yield ctx.call_activity(synthesize_results,
        input={"topic": topic, "research_results": research_results})

    return final_report


@task(description="Generate 3 focused research questions about {topic}.")
def generate_questions(topic: str) -> Questions:
    """Generates three research questions related to the given topic."""
    pass


@task(
    description="Research information to answer this question: {question}. Provide a detailed response.")
def gather_information(question: str) -> str:
    """Fetches relevant information based on the research question provided."""
    pass


@task(
    description="Create a comprehensive research report on {topic} based on the following research: {research_results}")
def synthesize_results(topic: str, research_results: List[str]) -> str:
    """Synthesizes the gathered research into a structured report."""
    pass


if __name__ == "__main__":
    wfapp = WorkflowApp()

    research_topic = "The environmental impact of quantum computing"

    logging.info(f"Starting research workflow on: {research_topic}")
    results = wfapp.run_and_monitor_workflow_sync(research_workflow, input=research_topic)
    logging.info(f"\nResearch Report:\n{results}")
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
dapr run --app-id dapr-agent-research --resources-path components/ -- python parallel_workflow.py
```
<!-- END_STEP -->

**How it works:**
This fan-out/fan-in pattern combines sequential and parallel execution:
1. First, `generate_questions()` executes sequentially
2. Multiple `gather_information()` tasks run in parallel using `ctx.when_all()`
3. The workflow waits for all parallel tasks to complete
4. Finally, `synthesize_results()` executes with all gathered data

## Additional Workflow Patterns

Beyond the patterns demonstrated in the examples, Dapr Agents supports other workflow patterns:

### Monitor Pattern

The **Monitor Pattern** periodically executes tasks in a loop at specified intervals. It's useful for scheduled jobs, polling external systems, or resource cleanup operations.

### External System Interaction Pattern

The **External System Interaction Pattern** allows workflows to wait for events from external systems before continuing. This pattern is ideal for approval workflows, integrating with external services, or implementing human-in-the-loop processes.

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

After completing this quickstart, move on to the [Multi-Agent Workflow quickstart](../05-multi-agent-workflow-dapr-workflows/README.md) to learn how to create distributed systems of collaborating agents.