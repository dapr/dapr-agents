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
