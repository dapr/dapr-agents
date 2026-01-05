import time

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext, WorkflowRuntime
from dotenv import load_dotenv

from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.decorators import llm_activity

load_dotenv()

# Initialize workflow runtime + LLM client
wfr = WorkflowRuntime()
llm = DaprChatClient(component_name="llm-provider")


@wfr.workflow(name="analyze_topic")
def analyze_topic(ctx: DaprWorkflowContext, topic: str):
    # Each step is durable and can be retried
    outline = yield ctx.call_activity(create_outline, input=topic)
    if len(outline) > 0:
        print("Outline:", outline, flush=True)
    blog_post = yield ctx.call_activity(write_blog, input=outline)
    if len(blog_post) > 0:
        print("Blog post:", blog_post, flush=True)
    return blog_post


@wfr.activity(name="create_outline")
@llm_activity(
    prompt="Create a very short outline about the topic '{topic}'. Provide 5 bullet points only.",
    llm=llm,
)
def create_outline(ctx, topic: str) -> str:
    # The llm_activity decorator handles the actual LLM invocation.
    pass


@wfr.activity(name="write_blog")
@llm_activity(
    prompt="Write a short (2 paragraphs) friendly blog post following this outline:\n{outline}",
    llm=llm,
)
def write_blog(ctx, outline: str) -> str:
    pass


if __name__ == "__main__":
    wfr.start()
    time.sleep(5)  # give the runtime time to initialize

    client = wf.DaprWorkflowClient()
    topic = "AI Agents"
    instance_id = client.schedule_new_workflow(
        workflow=analyze_topic,
        input=topic,
    )
    print(f"Workflow started: {instance_id}", flush=True)

    state = client.wait_for_workflow_completion(instance_id)
    if not state:
        print("No workflow state returned.")
    elif state.runtime_status.name == "COMPLETED":
        print(f"\n✅ Final Blog Post:\n{state.serialized_output}")
    else:
        print(f"Workflow finished with status: {state.runtime_status}")
        if state.failure_details:
            fd = state.failure_details
            print("Failure type:", fd.error_type)
            print("Failure message:", fd.message)
            print("Stack trace:\n", fd.stack_trace)

    wfr.shutdown()
