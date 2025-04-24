import dapr.ext.workflow as wf
from dotenv import load_dotenv
from openai import OpenAI
from time import sleep

# Load environment variables
load_dotenv()

# Initialize Workflow Instance
wfr = wf.WorkflowRuntime()


# Define Workflow logic
@wfr.workflow(name="lotr_workflow")
def task_chain_workflow(ctx: wf.DaprWorkflowContext):
    result1 = yield ctx.call_activity(get_character)
    result2 = yield ctx.call_activity(get_line, input=result1)
    return result2


# Activity 1
@wfr.activity(name="step1")
def get_character(ctx):
    client = OpenAI()
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Pick a random character from The Lord of the Rings and respond with the character name only",
            }
        ],
        model="gpt-4o",
    )
    character = response.choices[0].message.content
    print(f"Character: {character}")
    return character


# Activity 2
@wfr.activity(name="step2")
def get_line(ctx, character: str):
    client = OpenAI()
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": f"What is a famous line by {character}"}],
        model="gpt-4o",
    )
    line = response.choices[0].message.content
    print(f"Line: {line}")
    return line


if __name__ == "__main__":
    wfr.start()
    sleep(5)  # wait for workflow runtime to start

    wf_client = wf.DaprWorkflowClient()
    instance_id = wf_client.schedule_new_workflow(workflow=task_chain_workflow)
    print(f"Workflow started. Instance ID: {instance_id}")
    state = wf_client.wait_for_workflow_completion(instance_id)
    print(f"Workflow completed! Status: {state.runtime_status}")

    wfr.shutdown()
