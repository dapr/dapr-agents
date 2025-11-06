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
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.zipkin.json import ZipkinExporter
    from opentelemetry.sdk.resources import Resource
    from dapr_agents.observability import DaprAgentsInstrumentor

    # Define the service name using a Resource
    resource = Resource(attributes={"service.name": "dapr-weather-agents"})

    # Set up the OpenTelemetry TracerProvider with the resource
    tracer_provider = TracerProvider(resource=resource)

    # Configure the Zipkin exporter (no service_name argument here)
    zipkin_exporter = ZipkinExporter(
        endpoint="http://localhost:9411/api/v2/spans"  # default Zipkin endpoint
    )

    # Attach the exporter to the tracer provider
    span_processor = BatchSpanProcessor(zipkin_exporter)
    tracer_provider.add_span_processor(span_processor)

    # Register the tracer provider globally
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
