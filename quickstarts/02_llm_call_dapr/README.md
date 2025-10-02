# LLM Call with Dapr Chat Client

This quickstart demonstrates how to use Dapr Agents' `DaprChatClient` to call LLMs. You'll learn how to configure [different LLM backends](https://docs.dapr.io/reference/components-reference/supported-conversation/) using Dapr components and switch between them without changing your application code.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key (for the OpenAI example)
- Dapr CLI installed

## Environment Setup

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

## Examples

### 1. Using the Echo Component

First, run the application using the echo component, which returns the received prompt. This component is useful for local development and testing, as it eliminates the need for an API token or network interaction with a real LLM.

Create a `.env` file in the project root with the following content.

```env
DAPR_LLM_COMPONENT_DEFAULT=echo
```

The OpenAI API key is not needed for this example.
Create a `echo.yaml` file in the component folder

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: echo
spec:
  type: conversation.echo
  version: v1
```

Run the basic text completion example:

<!-- STEP
name: Run text completion example
expected_stdout_lines:
  - "Response:"
  - "Response with prompty:"
  - "Response with user input:"
timeout_seconds: 30
output_match_mode: substring
-->
```bash
dapr run --app-id dapr-llm --resources-path ./components -- python text_completion.py
```
<!-- END_STEP -->

The script uses the `DaprChatClient` which connects to Dapr's `echo` LLM component:

```python
from dotenv import load_dotenv

from dapr_agents.llm import DaprChatClient
from dapr_agents.types import LLMChatResponse, UserMessage

# Load environment variables from .env
load_dotenv()

# Basic chat completion
llm = DaprChatClient()
response: LLMChatResponse = llm.generate("Name a famous dog!")

if response.get_message() is not None:
    print("Response: ", response.get_message().content)

# Chat completion using a prompty file for context
llm = DaprChatClient.from_prompty("basic.prompty")
response: LLMChatResponse = llm.generate(input_data={"question": "What is your name?"})

if response.get_message() is not None:
    print("Response with prompty: ", response.get_message().content)

# Chat completion with user input
llm = DaprChatClient()
response: LLMChatResponse = llm.generate(messages=[UserMessage("hello")])

if response.get_message() is not None and "hello" in response.get_message().content.lower():
    print("Response with user input: ", response.get_message().content)
```

**Expected output:** The echo component will simply return the prompts that were sent to it.

**How It Works:**
1. Dapr starts, loading all resources from the `components` folder.
2. The client application retrieves the `DAPR_LLM_COMPONENT_DEFAULT` environment variable and uses it to communicate with Dapr's `echo` component.
3. Dapr's `echo` component, simply returns back the input it receives.
4. The application prints the output, which matches the input.

### 2. Switching to the OpenAI

Now, let's switch to using OpenAI by changing just the environment variable in the `.env` file:

```env
DAPR_LLM_COMPONENT_DEFAULT=openai
```

The OpenAI component configuration is in `components/openai.yaml`. You have two options to configure your API key:

### Option 1: Using Environment Variables (Recommended)

1. Create a `.env` file in the project root and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

2. When running the examples with Dapr, use the helper script to resolve environment variables:
```bash
# Get the environment variables from the .env file:
export $(grep -v '^#' ../../.env | xargs)

# Create a temporary resources folder with resolved environment variables
temp_resources_folder=$(../resolve_env_templates.py ./components)

# Run your dapr command with the temporary resources
dapr run --app-id dapr-llm --resources-path $temp_resources_folder -- python text_completion.py

# Clean up when done
rm -rf $temp_resources_folder
```

Note: The temporary resources folder will be automatically deleted when the Dapr sidecar is stopped or when the computer is restarted.

### Option 2: Direct Component Configuration

You can directly update the `key` in [components/openai.yaml](components/openai.yaml):
```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: openai
spec:
  type: conversation.openai
  metadata:
    - name: key
      value: "YOUR_OPENAI_API_KEY"
    - name: model
      value: gpt-4-turbo
    - name: cacheTTL
      value: 10m
```

Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.

Note: Many LLM providers are compatible with OpenAI's API (DeepSeek, Google AI, etc.) and can be used with this component by configuring the appropriate parameters. Dapr also has [native support](https://docs.dapr.io/reference/components-reference/supported-conversation/) for other providers like Google AI, Anthropic, Mistral, DeepSeek, etc.

Run the application the same way as before:

```bash
dapr run --app-id dapr-llm --resources-path components/ -- python text_completion.py
```

**Expected output:** The OpenAI component will respond with a different reply to each prompt.

**How It Works:**
1. Dapr starts, loading all resources from the `components` folder.
2. The client application retrieves the `DAPR_LLM_COMPONENT_DEFAULT` environment variable and uses it to communicate with Dapr's `openai` component.
3. The component defined in `components/openai.yaml` talks to OpenAI APIs.
4. The application prints the output returned from OpenAI

### 3. Simulating Failure with AWS BedRock

To demonstrate `DaprChatClient`'s resiliency features, let's create an example that will intentionally fail. We'll configure an AWS Bedrock component that won't be able to connect to a real service from the local machine, then show how Dapr's resiliency policies attempt to recover.

First, set the environment variable to use the AWS Bedrock component:

```python
DAPR_LLM_COMPONENT_DEFAULT=awsbedrock
```

Create an AWS Bedrock component configuration in `components/awsbedrock.yaml`:

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: awsbedrock
spec:
  type: conversation.aws.bedrock
  metadata:
    - name: endpoint
      value: "http://localhost:4566"
    - name: model
      value: amazon.titan-text-express-v1
    - name: cacheTTL
      value: 10m
```

This component is configured to connect to a local endpoint that simulates AWS services. It will intentionally fail since we don't have a real AWS Bedrock service running locally.

Configure resiliency policies in `components/resiliency.yaml`:

```yaml
apiVersion: dapr.io/v1alpha1
kind: Resiliency
metadata:
  name: awsbedrock-resiliency
spec:
  policies:
    timeouts:
      short-timeout: 1s
    retries:
      fixed-retry:
        policy: constant
        duration: 1s
        maxRetries: 3
  targets:
    components:
      awsbedrock:
        outbound:
          timeout: short-timeout
          retry: fixed-retry
```

This resiliency configuration applies only to the `awsbedrock` component and sets a timeout after 1 second, attempts it for  3 times , each with a 1-second delay between them.

Run the application the same way as before:

```bash
dapr run --app-id dapr-llm --resources-path components/ -- python text_completion.py
```

When you run this, you'll see output showing Dapr's retry mechanism in action:

```
WARN[0002] Error processing operation component[awsbedrock] output. Retrying in 1s...
== APP == 	details = "...exceeded maximum number of attempts, 3, https response error ... connect: connection refused"
```

This demonstrates how Dapr connection attempt times out in 1s, then attempts to retry the operation according to the policy (3 times with 1-second intervals) before finally failing with an error.

## Key Concepts

- **DaprChatClient**: A client that communicates with LLMs through Dapr's [Conversation API](https://docs.dapr.io/developing-applications/building-blocks/conversation/)
- **Dapr Components**: Pluggable building blocks that provide LLM capabilities
- **Environment Configuration**: Using environment variables to control component selection
- **Separation of Concerns**: Application logic separated from LLM provider implementation and operational concerns.

## Advantages of Using DaprChatClient

1. **Provider Agnostic**: Write code once and switch between different LLM providers.
2. **Prompt Caching** – Reducing latency and costs by storing and reusing repetitive prompts across API calls, by leveraging local caching.
3. **Personally Identifiable Information (PII) Obfuscation** – Automatically detect and mask sensitive user information from inputs and outputs.
4. **Secret Management**: Handle API keys securely through Dapr's [secret stores](https://docs.dapr.io/reference/components-reference/supported-secret-stores/)
5. **Resilience Patterns**: Benefit from Dapr's built-in timeout, retry and circuit-breaking [capabilities](https://docs.dapr.io/operations/resiliency/resiliency-overview/)
6. **Simplified Testing**: Use the echo component during development and testing

By using Dapr components for LLM interactions, you gain flexibility, modularity, and separation of concerns that make your application more maintainable and adaptable to changing requirements or LLM providers.

## Troubleshooting

1. **Environment Variable Issues**: Check that `DAPR_LLM_COMPONENT_DEFAULT` is set correctly
2. **Component Not Found**: Ensure the component yaml files are in the `components` directory
3. **Authentication Errors**: Verify your OpenAI API key is correctly set in the `components/openai.yaml` file

## Next Steps

After completing these examples, you could:

1. Interact with other LLMs supported by Dapr [Conversation Components](https://docs.dapr.io/reference/components-reference/supported-conversation/).
2. Implement structured outputs using `DaprChatClient` and Pydantic models
3. Move on to the [Agent Tool Call quickstart](../03-agent-tool-call) to learn how to build agents that can use tools
