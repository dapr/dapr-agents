from dapr_agents.workflow import WorkflowApp, workflow, task
from dapr.ext.workflow import DaprWorkflowContext
from dapr_agents import Agent
from dapr_agents.types import AssistantMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define simple agents
extractor = Agent(
    name="DestinationExtractor",
    role="Extract destination",
    instructions=["Extract the main city from the user query"],
)

planner = Agent(
    name="PlannerAgent",
    role="Outline planner",
    instructions=["Generate a 3-day outline for the destination"],
)

expander = Agent(
    name="ItineraryAgent",
    role="Itinerary expander",
    instructions=["Expand the outline into a detailed plan"],
)


# Define tasks
@task(agent=extractor)
def extract() -> AssistantMessage:
    pass


@task(agent=planner)
def plan() -> AssistantMessage:
    pass


@task(agent=expander)
def expand() -> AssistantMessage:
    pass


# Orchestration
@workflow(name="chained_planner_workflow")
def chained_planner_workflow(ctx: DaprWorkflowContext, user_msg: str):
    dest = yield ctx.call_activity(extract, input=user_msg)
    outline = yield ctx.call_activity(plan, input=dest["content"])
    itinerary = yield ctx.call_activity(expand, input=outline["content"])
    return itinerary["content"]


if __name__ == "__main__":
    wfapp = WorkflowApp()

    results = wfapp.run_and_monitor_workflow_sync(
        chained_planner_workflow, input="Plan a trip to Paris"
    )
    print(f"Trip Itinerary: {results}")
