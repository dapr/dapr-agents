import asyncio

from dapr_agents.llm import DaprChatClient

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import AgentMemoryConfig, AgentStateConfig
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.observability import DaprAgentsInstrumentor
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner
from function_tools import slow_weather_func
from opentelemetry import trace
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_tracing() -> TracerProvider:
    """Configure OpenTelemetry tracing with Zipkin and instrument Dapr Agents."""
    resource = Resource(attributes={"service.name": "dapr-durable-agent"})
    tracer_provider = TracerProvider(resource=resource)

    zipkin_exporter = ZipkinExporter(endpoint="http://localhost:9411/api/v2/spans")
    span_processor = BatchSpanProcessor(zipkin_exporter)
    tracer_provider.add_span_processor(span_processor)

    trace.set_tracer_provider(tracer_provider)

    instrumentor = DaprAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    return tracer_provider


async def main() -> None:
    tracer_provider = setup_tracing()

    weather_agent = DurableAgent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[slow_weather_func],
        llm=DaprChatClient(component_name="llm-provider"),
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="conversation-statestore",
                session_id="08-durable-agent-trace",
            )
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflow-statestore"),
        ),
    )

    runner = AgentRunner()
    try:
        prompt = "What is the weather in London?"
        result = await runner.run(weather_agent, payload={"task": prompt})
        print(f"Agent result: {result}")
    finally:
        runner.shutdown(weather_agent)
        tracer_provider.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
