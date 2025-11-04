from __future__ import annotations

from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.decorators import llm_activity
from dapr_agents.workflow.decorators.routers import message_router

load_dotenv()


class StartBlogMessage(BaseModel):
    topic: str = Field(min_length=1, description="Blog topic/title")


# Initialize the LLM client and workflow runtime
llm = DaprChatClient(component_name="openai")


@message_router(
    pubsub="messagepubsub", topic="blog.requests", message_model=StartBlogMessage
)
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
