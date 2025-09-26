# Durable Agent Tool Call with Dapr Conversation API

This quickstart mirrors `03-durable-agent-multitool-call/` but uses the Dapr Conversation API Alpha2 as the LLM provider with tool calling.
It tests the Durable Agent with multiple tools using the Dapr Conversation API Alpha2.

## Prerequisites
- Python 3.10+
- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
- Dapr CLI [installed and initialized](https://docs.dapr.io/getting-started/install-dapr-cli/#step-1-install-the-dapr-cli)

## Environment Setup

<details open>
<summary><strong>Option 1: Using uv (Recommended)</strong></summary>

<!-- STEP
name: Run DaprChatClient multitool example
expected_stderr_lines:
  - "Creating virtual environment"
expected_stdout_lines:
  - "What's the current weather in Boston, MA, then compute (14*7)+23, and finally search for the official tourism site for Boston?"
  - "Function name: GetWeather"
  - 'Arguments: {"location": "Boston, MA"}'
  - " Boston, MA: 85"
  - "Function name: Calculate"
  - 'Arguments: {"expression": "(14*7)+23"}'
  - "121.0"
  - "Function name: WebSearch"
  - 'Arguments: {"query": "official tourism site for Boston"'
timeout_seconds: 30
output_match_mode: substring
match_order: none
-->


```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install core dependencies
uv pip install -r requirements.txt
```

</details>

<details>
<summary><strong>Option 2: Using pip</strong></summary>

```text
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

</details>

## Component Configuration

We provide a `components` folder with the Dapr components for the LLM and state/pubsub.
The Conversation component example uses [OpenAI](https://docs.dapr.io/reference/components-reference/supported-conversation/openai/) component. 

Many other LLM providers are compatible with OpenAI to certain extent (DeepSeek, Google AI, etc) so you can use them with the OpenAI component by configuring it with the appropriate parameters.
But Dapr also has [native support](https://docs.dapr.io/reference/components-reference/supported-conversation/) for other providers like Google AI, Anthropic, Mistral, DeepSeek, etc.

One thing you will need to update is the `key` in the [component configuration file](components/openai.yaml). You can do it directly in the file:

```yaml
metadata:
  - name: key
    value: "YOUR_OPENAI_API_KEY"
```

Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API (or other provider's API) key.

## Run
```text
dapr run --app-id durablemultitoolapp \
  --resources-path ./components \
  -- python multi_tool_agent_dapr.py
```

Another safer option is to use the environment variable `OPENAI_API_KEY` and use the helper (quickstarts/resolve_env_templates.py) to render a temporary resources folder and pass it to Dapr:
```bash
# Get the environment variables from the .env file in the root directory:
export $(grep -v '^#' ../../.env | xargs)

# create the temporary resources folder
temp_resources_folder=$(../resolve_env_templates.py ./components)

# run the application
dapr run --resources-path $temp_resources_folder -- python multi_tool_agent_dapr.py

# delete the temporary resources folder
rm -rf $temp_resources_folder
```

<!-- END_STEP -->

The temporary resources folder will be deleted after the Dapr sidecar is stopped or when the computer is restarted.

## Files
- `multi_tool_agent_dapr.py`: Durable agent using `llm_provider="dapr"`
- `multi_tools.py`: sample tools
- `components/`: Dapr components for LLM and state/pubsub

Notes:
- Alpha2 currently does not support streaming; this example is non-streaming.
