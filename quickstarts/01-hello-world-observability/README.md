# Hello World with Dapr Agents and Tracing

This quickstart provides a hands-on introduction to setting up full end-to-end tracing with Dapr Agents.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key
- Zipkin
- Jaeger

## Environment Setup

### Option 1: Using pip (Recommended)

```bash
# Create a virtual environment
python3.10 -m venv .venv

# Activate the virtual environment 
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### Option 2: Using uv 

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install core dependencies
uv pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

## Examples

### 1. Simple Agent with Zipkin tracing

In this example, we'll run a simple agent that uses Zipkin for distributed tracing.

Run Zipkin locally:

```bash
docker run --rm -d -p 9411:9411 --name zipkin openzipkin/zipkin
```

Run the agent example to see how to create an agent with custom tools:

<!-- STEP
name: Run simple agent with tools example
expected_stdout_lines:
  - "user:"
  - "What's the weather?"
  - "assistant:"
  - "Function name: MyWeatherFunc"
  - "MyWeatherFunc(tool)"
  - "It's 72°F and sunny"
  - "assistant:"
  - "The current weather is 72°F and sunny."
timeout_seconds: 30
output_match_mode: substring
-->
```bash
python 01_agent_zipkin.py
```
<!-- END_STEP -->

**Expected output:** Visit `http://localhost:9411` in your browser and view the traces.

### 2. Simple Agent with OpenTelemetry tracing (Jaeger)

In this example, we'll use Jaeger as the tracing backend with an HTTP `OTLPSpanExporter` generic span exporter.

Run Jaeger locally:

```bash
docker run -d -e COLLECTOR_OTLP_ENABLED=true -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one:latest
```

Run the agent example to see how to create an agent with custom tools:

<!-- STEP
name: Run simple agent with tools example
expected_stdout_lines:
  - "user:"
  - "What's the weather?"
  - "assistant:"
  - "Function name: MyWeatherFunc"
  - "MyWeatherFunc(tool)"
  - "It's 72°F and sunny"
  - "assistant:"
  - "The current weather is 72°F and sunny."
timeout_seconds: 30
output_match_mode: substring
-->
```bash
python 02_agent_otel.py
```
<!-- END_STEP -->

**Expected output:** Visit `http://localhost:16686` in your browser and view the traces.

## Next Steps

After completing these examples, move on to the [LLM Call quickstart](../02_llm_call_open_ai/README.md) to learn more about structured outputs from LLMs.
