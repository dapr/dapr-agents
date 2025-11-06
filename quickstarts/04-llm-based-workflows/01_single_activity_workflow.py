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
