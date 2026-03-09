#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.decorators.decorators import message_router

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


async def create_outline(ctx, topic: str) -> str:
    return str(
        llm.generate(
            prompt=f"Create a short outline about {topic}. Output 3-5 bullet points."
        )
    )


async def write_post(ctx, outline: str) -> str:
    return str(
        llm.generate(
            prompt=f"Write a short blog post following this outline:\n{outline}"
        )
    )
