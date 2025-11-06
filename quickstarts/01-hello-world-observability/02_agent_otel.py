import asyncio
from dapr_agents import tool, Agent, OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()


@tool
def my_weather_func() -> str:
    """Get current weather."""
    return "It's 72°F and sunny"


async def main():
    weather_agent = Agent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[my_weather_func],
        llm=OpenAIChatClient(model="gpt-3.5-turbo"),
    )

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from dapr_agents.observability import DaprAgentsInstrumentor

    # Define the service name in the resource
    resource = Resource(attributes={"service.name": "dapr-weather-agents"})

    # Set up TracerProvider with resource
    tracer_provider = TracerProvider(resource=resource)

    # Set up OTLP exporter (in this example working with HTTP to send traces to Jaeger)
    otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")

    # Set up span processor and add to tracer provider
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    # Register tracer provider globally
    trace.set_tracer_provider(tracer_provider)

    # Instrument Dapr Agents
    instrumentor = DaprAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    # Run the agent
    try:
        await weather_agent.run(
            "What is the weather in Virginia, New York and Washington DC?"
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
