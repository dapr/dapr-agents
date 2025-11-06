from __future__ import annotations

import logging

import dapr.ext.workflow as wf
from dapr.clients.grpc._response import TopicEventResponse
from pydantic import BaseModel, Field

from dapr_agents.workflow.decorators.routers import message_router

logger = logging.getLogger(__name__)


class StartBlogMessage(BaseModel):
    topic: str = Field(min_length=1, description="Blog topic/title")


# Import the workflow after defining models to avoid circular import surprises
from workflow import blog_workflow  # noqa: E402


@message_router(pubsub="messagepubsub", topic="blog.requests")
def start_blog_workflow(message: StartBlogMessage) -> TopicEventResponse:
    """
    Triggered by pub/sub. Validates payload via Pydantic and schedules the workflow.
    """
    try:
        client = wf.DaprWorkflowClient()
        instance_id = client.schedule_new_workflow(
            workflow=blog_workflow,
            input=message.model_dump(),
        )
        logger.info(
            "Scheduled blog_workflow instance=%s topic=%s", instance_id, message.topic
        )
        return TopicEventResponse("success")
    except Exception as exc:  # transient infra error → retry
        logger.exception("Failed to schedule blog workflow: %s", exc)
        return TopicEventResponse("retry")
