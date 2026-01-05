import time

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext, WorkflowRuntime
from dotenv import load_dotenv

from dapr_agents import Agent, tool
from dapr_agents.agents.configs import AgentMemoryConfig
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.workflow.decorators import agent_activity

load_dotenv()

# Initialize workflow runtime + LLM client
wfr = WorkflowRuntime()
llm = DaprChatClient(component_name="llm-provider")


# ------------- TOOLS -------------
@tool
def get_customer_info(customer_name: str) -> str:
    """Get customer information by name. Returns a simple text description."""
    # Simple mock customer data
    customers = {
        "alice": "Customer: Alice, Premium Plan, 5 active services",
        "bob": "Customer: Bob, Standard Plan, 2 active services",
        "charlie": "Customer: Charlie, Basic Plan, 1 active service",
    }
    return customers.get(
        customer_name.lower(),
        f"Customer: {customer_name}, Standard Plan, 1 active service",
    )


# ------------- AGENTS -------------
triage_agent = Agent(
    name="Triage Agent",
    role="Customer Support Triage Assistant",
    goal="Gather customer information and prepare a triage summary.",
    instructions=[
        "Use the tool to get customer information, then combine it with the issue description.",
    ],
    llm=llm,
    tools=[get_customer_info],
    memory=AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="conversation-statestore",
            session_id="04-support-triage",
        )
    ),
)

expert_agent = Agent(
    name="Expert Agent",
    role="Technical Support Specialist",
    goal="Provide recommendations based on customer context and issue.",
    instructions=[
        "Provide a clear, actionable recommendation to resolve the issue.",
    ],
    llm=llm,
    memory=AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="conversation-statestore",
            session_id="04-support-expert",
        )
    ),
)


# ------------- WORKFLOW -------------
@wfr.workflow(name="support_workflow")
def support_workflow(ctx: DaprWorkflowContext, request: dict):
    """Process a support request through triage and expert agents."""
    # Each step is durable and can be retried
    triage_result = yield ctx.call_activity(triage_request, input=request)
    if triage_result:
        print("Triage result:", triage_result.get("content", ""), flush=True)

    recommendation = yield ctx.call_activity(
        get_recommendation, input=triage_result.get("content", "")
    )
    if recommendation:
        print("Recommendation:", recommendation.get("content", ""), flush=True)

    return recommendation.get("content", "") if recommendation else ""


# ------------- ACTIVITIES -------------
@wfr.activity(name="triage_request")
@agent_activity(agent=triage_agent)
def triage_request(ctx, customer: str, issue: str) -> dict:
    """Triage the support request by gathering customer info and summarizing.

    The workflow passes a dict with `customer` and `issue` keys, which map to these parameters.
    """
    pass


@wfr.activity(name="get_recommendation")
@agent_activity(agent=expert_agent)
def get_recommendation(ctx) -> dict:
    """Get expert recommendation based on triage summary."""
    pass


if __name__ == "__main__":
    wfr.start()
    time.sleep(5)  # give the runtime time to initialize

    client = wf.DaprWorkflowClient()
    request = {
        "customer": "alice",
        "issue": "Unable to access dashboard after recent update",
    }
    instance_id = client.schedule_new_workflow(
        workflow=support_workflow,
        input=request,
    )
    print(f"Workflow started: {instance_id}", flush=True)

    state = client.wait_for_workflow_completion(instance_id)
    if not state:
        print("No workflow state returned.")
    elif state.runtime_status.name == "COMPLETED":
        print(f"\n✅ Final Recommendation:\n{state.serialized_output}")
    else:
        print(f"Workflow finished with status: {state.runtime_status}")
        if state.failure_details:
            fd = state.failure_details
            print("Failure type:", fd.error_type)
            print("Failure message:", fd.message)
            print("Stack trace:\n", fd.stack_trace)

    wfr.shutdown()
