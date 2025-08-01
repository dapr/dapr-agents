from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv
from phoenix.otel import register

from dapr_agents import Agent, OpenAIChatClient, NVIDIAChatClient, HFHubChatClient
from dapr_agents.observability import DaprAgentsInstrumentor
from dapr_agents.types import AssistantMessage
from dapr_agents.workflow import WorkflowApp, task, workflow

# Load environment variables
load_dotenv()

# Register Dapr Agents with Phoenix OpenTelemetry
tracer_provider = register(
    project_name="dapr-weather-agents",
    protocol="http/protobuf",
)
# Initialize Dapr Agents OpenTelemetry instrumentor
instrumentor = DaprAgentsInstrumentor()
instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

# Define simple agents
openai_llm = OpenAIChatClient(model="gpt-4o-mini")
nvidia_llm = NVIDIAChatClient(model="meta/llama-3.1-8b-instruct")
hf_llm = HFHubChatClient(model="HuggingFaceTB/SmolLM3-3B")

extractor = Agent(
    name="DestinationExtractor",
    role="Extract destination",
    instructions=["Extract the main city from the user query"],
    llm=openai_llm,  # Use OpenAI LLM for extraction
)

planner = Agent(
    name="PlannerAgent",
    role="Outline planner",
    instructions=["Generate a 3-day outline for the destination"],
    llm=nvidia_llm,  # Use NVIDIA LLM for planning
)

expander = Agent(
    name="ItineraryAgent",
    role="Itinerary expander",
    instructions=["Expand the outline into a detailed plan"],
    llm=hf_llm,  # Use Hugging Face LLM for expansion
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
