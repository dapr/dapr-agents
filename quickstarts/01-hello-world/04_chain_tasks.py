from floki import WorkflowApp
from floki.types import DaprWorkflowContext

wfapp = WorkflowApp()

@wfapp.workflow(name='research_workflow')
def analyze_topic(ctx: DaprWorkflowContext, topic: str):
    # Each step is durable and can be retried
    outline = yield ctx.call_activity(create_outline, input=topic)
    blog_post = yield ctx.call_activity(write_blog, input=outline)
    return blog_post

@wfapp.task(description="Create a detailed outline about {topic}")
def create_outline(topic: str) -> str:
    pass

@wfapp.task(description="Write a comprehensive blog post following this outline: {outline}")
def write_blog(outline: str) -> str:
    pass

if __name__ == '__main__':
    results = wfapp.run_and_monitor_workflow(
        workflow=analyze_topic,
        input="AI Agents"
    )