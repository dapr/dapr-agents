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

"""
Example 2 — HITL via Pub/Sub.

Agent side : when the workflow pauses, it publishes an ApprovalRequiredEvent to
             the Dapr pub/sub topic configured in AgentApprovalConfig.
Client side: a Dapr subscriber receives that event, inspects it, and publishes
             an ApprovalResponseEvent back to a response topic. The agent's
             pub/sub subscriber picks it up and raises the workflow event to resume.

Both agent and client use Dapr pub/sub throughout — no direct HTTP calls or
workflow client calls on the client side.

Prerequisites
-------------
- Dapr sidecar running  (dapr run ...)
- A Dapr pub/sub component named "pubsub" (e.g. Redis streams)
- Two topics: "agent-approval-requests" (outbound) and "agent-approval-responses" (inbound)

Other patterns:
  - durable_agent_hitl.py  — approval round-trip over HTTP
  - hitl_wf_event.py       — approval round-trip via direct workflow event
"""

import asyncio
import json
import logging

from dapr.aio.clients import DaprClient
from dotenv import load_dotenv

from dapr_agents import DurableAgent, tool
from dapr_agents.agents.configs import AgentApprovalConfig, AgentExecutionConfig
from dapr_agents.agents.schemas import ApprovalRequiredEvent, ApprovalResponseEvent
from dapr_agents.hooks import Hooks, HookContext, HookDecision, RequireApproval, Proceed
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.llm.openai import OpenAIChatClient

load_dotenv()

logger = logging.getLogger(__name__)

PUBSUB_NAME = "pubsub"
REQUEST_TOPIC = "agent-approval-requests"
RESPONSE_TOPIC = "agent-approval-responses"
APPROVED = True  # set to False to test the denial path


@tool
def delete_old_data(dataset: str) -> str:
    """Permanently delete a dataset by name."""
    return f"Dataset '{dataset}' has been deleted."


def before_tool(ctx: HookContext) -> HookDecision:
    if ctx.step_name == "DeleteOldData":
        return RequireApproval(
            timeout_seconds=300,
            instructions=f"Confirm deletion of dataset: {ctx.payload.get('dataset')}",
        )
    return Proceed()


async def approval_subscriber(agent: DurableAgent) -> None:
    """
    Client side: subscribe to REQUEST_TOPIC, read the ApprovalRequiredEvent,
    and publish an ApprovalResponseEvent back to RESPONSE_TOPIC.

    In a real system this would be a separate service (a Slack bot, a dashboard,
    etc.). Here it runs as a background coroutine in the same process so the
    example is self-contained.
    """
    async with DaprClient() as client:
        # Dapr Python SDK streaming subscription
        async with client.subscribe(
            pubsub_name=PUBSUB_NAME,
            topic=REQUEST_TOPIC,
        ) as subscription:
            async for message in subscription:
                try:
                    event = ApprovalRequiredEvent(**json.loads(message.data()))
                except Exception:
                    logger.exception("Failed to parse ApprovalRequiredEvent; skipping.")
                    await subscription.respond_success(message)
                    continue

                logger.info(
                    f"Subscriber received approval request: tool='{event.step_name}' "
                    f"request_id={event.approval_request_id}"
                )

                # Publish the human decision back via pub/sub.
                response = ApprovalResponseEvent(
                    approval_request_id=event.approval_request_id,
                    approved=APPROVED,
                    reason="approved via pub/sub example",
                )
                await client.publish_event(
                    pubsub_name=PUBSUB_NAME,
                    topic_name=RESPONSE_TOPIC,
                    data=response.model_dump_json(),
                    data_content_type="application/json",
                )
                logger.info(
                    f"Subscriber published response: approved={APPROVED} "
                    f"for request_id={event.approval_request_id}"
                )
                await subscription.respond_success(message)


async def main():
    logging.basicConfig(level=logging.INFO)

    agent = DurableAgent(
        name="hitl-pubsub-demo",
        role="Operations Assistant",
        goal="Help the user manage data.",
        instructions=["Use delete_old_data when asked to clean up or remove data."],
        llm=OpenAIChatClient(model="gpt-4o-mini"),
        tools=[delete_old_data],
        hooks=Hooks(before_tool_call=[before_tool]),
        execution=AgentExecutionConfig(
            approval=AgentApprovalConfig(
                pubsub_name=PUBSUB_NAME,
                topic=REQUEST_TOPIC,
            )
        ),
    )

    runner = AgentRunner()

    # Start the client-side subscriber as a background task.
    subscriber_task = asyncio.create_task(approval_subscriber(agent))

    instance_id = await runner.run(
        agent,
        payload={"task": "Please delete the old 'sales-2023' dataset."},
        wait=False,
    )
    print(f"\nWorkflow started: instance_id = {instance_id}")
    print(
        f"Agent will publish approval request to topic '{REQUEST_TOPIC}'.\n"
        f"Subscriber will respond on topic '{RESPONSE_TOPIC}'.\n"
    )

    result = await asyncio.to_thread(
        runner.wait_for_workflow_completion,
        instance_id,
        timeout_in_seconds=360,
    )
    if result:
        print(f"\nFinal output:\n{getattr(result, 'serialized_output', result)}\n")
    else:
        print("\nWorkflow did not complete within the wait window.")

    subscriber_task.cancel()
    runner.shutdown(agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
