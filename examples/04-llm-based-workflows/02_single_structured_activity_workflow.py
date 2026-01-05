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
    "name":  string,   // Dog\'s full name
    "bio":   string,   // 1-3 sentence biography
    "breed": string    // Primary breed or mixed
}}
""",
    llm=llm,
)
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
