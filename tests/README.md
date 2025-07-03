# Dapr Agents Test Framework

This directory contains the comprehensive test suite for the Dapr Agents project, covering unit tests, integration tests, and end-to-end scenarios across different agent types and streaming capabilities.

## Test Categories

### 1. Unit Tests (`tests/unit/`)
- **Purpose**: Fast, isolated tests with no external dependencies
- **Runtime**: Milliseconds
- **Dependencies**: None (pure Python logic)
- **Coverage**: Tool utilities, function calling, prompt processing, memory components

### 2. Integration Tests (`tests/integration/`)
- **Purpose**: Test interactions with Dapr runtime and components
- **Runtime**: Seconds to minutes
- **Dependencies**: Dapr runtime + components
- **Coverage**: Chat scenarios, React agents, AssistantAgent workflows

### 3. Manual Tests (`tests/manual/`)
- **Purpose**: Development helpers and debugging scripts
- **Runtime**: Variable
- **Dependencies**: Various
- **Coverage**: Component validation, provider testing, troubleshooting

## Infrastructure Requirements

### Core Infrastructure (Required for All Tests)

#### 1. Dapr Infrastructure Services
```bash
# Initialize Dapr with placement service, Redis, and Zipkin
dapr init
```

This provides:
- **Placement Service** (port 50005) - Required for Actor/Workflow runtime
- **Redis** (port 6379) - Required for state storage
- **Zipkin** (port 9411) - Optional for tracing
- **Scheduler** (port 50006) - For scheduled workflows

#### 2. Local Development Environment (if you are also making changes on the SDK and/or Dapr runtime)
- **Dapr Repository**: `../dapr` (for local development sidecar)
- **Python SDK**: `../python-sdk` (optional, for SDK development)
- **Components**: `./components/` (conversation components)

### Test-Specific Infrastructure

#### Basic Chat Tests (Echo, Anthropic, OpenAI, Gemini)
- ✅ Local development sidecar
- ✅ Conversation components
- ✅ API keys (optional for echo)

#### React Agent Tests
- ✅ Local development sidecar
- ✅ Conversation components
- ✅ Tool definitions

#### AssistantAgent Tests
- ✅ **All of the above PLUS:**
- ✅ Placement service connection
- ✅ Redis state stores (workflow, registry, conversation)
- ✅ Redis pub/sub (for multi-agent messaging)

## Environment Setup

### 1. Environment Variables
Create a `.env` file in the repository root:

```bash
# Required for real provider testing
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_AI_API_KEY=your_gemini_api_key_here
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here

# Optional for enhanced testing
ELEVENLABS_API_KEY=your_elevenlabs_key_here
NVIDIA_API_KEY=your_nvidia_key_here
```

### 2. Infrastructure Initialization
```bash
# 1. Initialize Dapr infrastructure
dapr init

# 2. Verify containers are running
docker ps
# Should show: dapr_placement, dapr_redis, dapr_zipkin, dapr_scheduler

# 3. Verify local Dapr build (for development)
ls ../dapr/dist/darwin_arm64/release/daprd  # or linux_amd64 for Linux
```

## Test Execution

### Quick Test Suite (Unit + Echo Integration)
```bash
# Run fast tests with no API keys required
cd tests
python -m pytest unit/ integration/test_basic_integration.py -v
```

### Complete Integration Tests
```bash
# Run all integration tests (requires API keys)
cd tests
python -m pytest integration/ -v
```

### Scenario-Specific Tests

#### 1. Basic Chat Scenarios
```bash
# Test all four core scenarios: non-streaming, streaming, tool calling, streaming + tool calling
python -m pytest integration/test_chat_scenarios.py -v -s
```

#### 2. React Agent Tests
```bash
# Test React agent patterns with tool calling
python -m pytest integration/test_react_agent_scenarios.py -v -s
```

#### 3. AssistantAgent Tests
```bash
# Test workflow-based agents (requires full infrastructure)
python -m pytest integration/test_assistant_agent_scenarios.py -v -s
```

### Provider-Specific Testing

#### Echo Provider
- **No API Key Required**: Perfect for development and CI/CD
- **All Scenarios Supported**: Basic chat, streaming, tool calling, streaming + tool calling
- **Behavior**: Echoes back requests with OpenAI-compatible response format
- **Tool Calling**: Simulates tool calls without actual execution

#### Anthropic Provider (Claude)
- **API Key Required**: `ANTHROPIC_API_KEY`
- **All Scenarios Supported**: Excellent streaming and tool calling support
- **Response Format**: May return multiple outputs (explanation + tool calls)
- **Streaming**: Works perfectly with real-time token streaming

#### OpenAI Provider (GPT)
- **API Key Required**: `OPENAI_API_KEY`
- **Standard Reference**: Most compatible format across scenarios
- **Tool Calling**: Full OpenAI tool calling specification support

#### Google AI Provider (Gemini)
- **API Key Required**: `GOOGLE_AI_API_KEY`
- **Provider-Specific Behavior**: Based on components-contrib conformance tests:
  - **Tool Call Metadata**: Google AI through langchaingo doesn't populate `Type` and `ID` fields like OpenAI
  - **Functional Limitation**: This is a langchaingo implementation detail, not a functional limitation
  - **Tool Calling Status**: ✅ **FULLY WORKING** - function name and arguments are properly populated
  - **Tool Result Handling**: Requires proper conversation flow with matching tool call IDs
  - **Testing Approach**: Cannot use isolated tool result tests; requires dynamic tool call ID extraction

#### Testing Strategy by Provider

```bash
# Echo - Development and CI/CD (no API key needed)
pytest tests/integration/test_chat_scenarios.py::TestChatScenarios::test_echo_scenarios -v

# Anthropic - Full feature validation (requires API key)
pytest tests/integration/test_chat_scenarios.py::TestChatScenarios::test_anthropic_scenarios -v

# Google AI - Provider-specific behavior validation (requires API key)  
pytest tests/integration/test_chat_scenarios.py::TestChatScenarios::test_gemini_scenarios -v

# All providers - Comprehensive validation
pytest tests/integration/test_chat_scenarios.py -v
```

## Development Workflow

### 1. Start Infrastructure for Testing

#### For Basic Chat and React Agent Tests
```bash
# Start local development sidecar (basic components)
python tools/run_dapr_dev.py --app-id test-app --components ./components --log-level info
```

#### For AssistantAgent Tests
```bash
# Start with placement service connection and custom ports
python tools/run_dapr_dev.py --app-id assistant-app --components ./components --port 3501 --grpc-port 50002 --log-level info
```

### 2. Run Tests in Another Terminal
```bash
cd tests

# Quick smoke test
python -m pytest integration/test_basic_integration.py::test_echo_conversation -v -s

# Full test suite
python -m pytest integration/ -v
```

### 3. Development Testing Loop
```bash
# 1. Make code changes
# 2. Run relevant tests
python -m pytest integration/test_chat_scenarios.py::TestChatScenarios::test_comprehensive_scenario_matrix -v -s

# 3. If AssistantAgent changes, test workflows
python -m pytest integration/test_assistant_agent_scenarios.py::TestAssistantAgentScenarios::test_assistant_agent_initialization -v -s
```

## Test Infrastructure Details

### Dapr Component Configuration

The test framework uses the following components:

#### Conversation Components
- **echo** - Basic testing (no API key)
- **echo-tools** - Tool calling testing (no API key)
- **anthropic** - Claude integration (requires `ANTHROPIC_API_KEY`)
- **gemini** - Gemini integration (requires `GOOGLE_AI_API_KEY`)
- **openai** - GPT integration (requires `OPENAI_API_KEY`)

#### State Management Components (AssistantAgent)
- **workflowstatestore** - Workflow state persistence
- **registrystatestore** - Agent discovery and registration
- **conversationstore** - Conversation memory
- **messagepubsub** - Multi-agent messaging

### Port Configuration

#### Standard Configuration (Basic Tests)
- **HTTP**: 3500
- **gRPC**: 50001
- **Metrics**: 9090

#### AssistantAgent Configuration (Workflow Tests)
- **HTTP**: 3501 (avoids conflicts)
- **gRPC**: 50002 (avoids conflicts)
- **Metrics**: 9091 (avoids conflicts)
- **Placement**: localhost:50005 (from `dapr init`)

## Test Scenarios Coverage

### ✅ Currently Tested

#### Basic Chat Scenarios
- **Non-streaming chat** - Direct conversation without tools
- **Streaming chat** - Real-time response streaming
- **Tool calling non-streaming** - Tool execution without streaming
- **Tool calling with streaming** - Tools + streaming combined

#### Agent Patterns
- **React Agents** - Reasoning-Action pattern with tools
- **AssistantAgent** - Workflow-based agents with state management

#### Provider Coverage
- **Echo** - All scenarios working
- **Anthropic** - Non-streaming working, streaming has known issues
- **OpenAI** - All scenarios should work (requires API key)
- **Gemini** - All scenarios should work (requires API key)

### 🔄 Provider-Specific Behavior Testing

The test framework validates provider-specific differences:
- **Response formats** (single vs multiple outputs)
- **Streaming compatibility** (OpenAI-compatible vs provider-specific)
- **Tool calling formats** (different providers may have different schemas)
- **Error handling** (provider-specific error responses)

## Troubleshooting

### Common Issues

#### 1. "Actor runtime disabled: placement service is not configured"
**Solution**: Ensure `dapr init` has been run and placement service is running
```bash
docker ps | grep placement
# Should show dapr_placement container
```

#### 2. "Port already in use" errors
**Solution**: Use custom ports for AssistantAgent tests
```bash
python tools/run_dapr_dev.py --port 3501 --grpc-port 50002
```

#### 3. "No API key" test failures
**Solution**: Either provide API keys in `.env` or skip those tests
```

## Local Development with Dapr Python SDK

### Why Use the SDK for Local Development?

When developing locally with all components (python-sdk, daprd sidecar, and dapr-agents) in the same environment, using the **Dapr Python SDK** provides significant advantages over raw HTTP requests:

#### 1. **Type Safety & Better API**
```python
# ✅ SDK - Type-safe with IDE support
from dapr.clients import DaprClient
client = DaprClient()
response = client.invoke_conversation(
    name="anthropic",
    inputs=[{"role": "user", "content": "Hello"}],
    tools=weather_tools,
    stream=True
)

# ❌ Raw HTTP - No type safety, manual serialization
import requests
response = requests.post(
    "http://localhost:3500/v1.0/conversation/anthropic",
    json={"inputs": [...], "tools": [...]}
)
```

#### 2. **Automatic Error Handling**
- **SDK**: Proper exception handling with `DaprInternalError`, `DaprGrpcError`
- **Raw HTTP**: Manual status code checking and error parsing
- **SDK**: Built-in retry logic and resilience patterns
- **Raw HTTP**: Must implement retry logic manually

#### 3. **Authentication & Security**
- **SDK**: Automatic token handling and mTLS support when configured
- **Raw HTTP**: Manual header management and certificate handling
- **SDK**: Proper credential management through Dapr client configuration
- **Raw HTTP**: Manual API key injection and security handling

#### 4. **Development Experience**
- **SDK**: IDE autocomplete, type hints, and error checking
- **Raw HTTP**: Manual URL construction and payload formatting
- **SDK**: Automatic serialization/deserialization of complex objects
- **Raw HTTP**: Manual JSON encoding/decoding with error-prone string manipulation

#### 5. **Local Development Integration**
```python
# ✅ SDK automatically detects local Dapr sidecar
client = DaprClient()  # Connects to localhost:50001 by default

# ✅ Streaming responses are properly handled
for chunk in client.invoke_conversation(name="anthropic", stream=True):
    print(chunk)  # Properly deserialized objects

# ✅ Tool calling with proper response objects
response = client.invoke_conversation(name="echo-tools", tools=tools)
for output in response.outputs:
    if hasattr(output, 'tool_calls'):
        for tool_call in output.tool_calls:
            print(f"Tool: {tool_call.function.name}")
```

### Local Development Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Test Suite    │    │  Dapr Python    │    │   Local Dapr    │
│                 │◄──►│      SDK        │◄──►│    Sidecar      │
│  (using SDK)    │    │                 │    │   (daprd)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   Conversation  │
                                              │   Components    │
                                              │ (anthropic,     │
                                              │  openai, etc.)  │
                                              └─────────────────┘
```

### SDK vs HTTP Comparison

| Aspect | Dapr Python SDK | Raw HTTP Requests |
|--------|----------------|-------------------|
| **Type Safety** | ✅ Full type hints | ❌ Manual typing |
| **Error Handling** | ✅ Structured exceptions | ❌ Manual parsing |
| **Streaming** | ✅ Iterator-based | ❌ Manual chunk processing |
| **Tool Calling** | ✅ Structured objects | ❌ Manual JSON parsing |
| **Authentication** | ✅ Automatic | ❌ Manual headers |
| **Local Development** | ✅ Auto-discovery | ❌ Manual URLs |
| **IDE Support** | ✅ Autocomplete | ❌ String-based |
| **Debugging** | ✅ Rich error info | ❌ HTTP status codes |
| **Maintainability** | ✅ High | ❌ Low |

### Example: Tool Calling with SDK

```python
from dapr.clients import DaprClient

# Define tools
weather_tools = [{
    "type": "function",
    "function": {
        "name": "GetWeather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

# Use SDK for tool calling
with DaprClient() as client:
    response = client.invoke_conversation(
        name="anthropic",
        inputs=[{
            "role": "user", 
            "content": "What's the weather in Boston?"
        }],
        tools=weather_tools
    )
    
    # Type-safe access to response
    for output in response.outputs:
        print(f"Result: {output.result}")
        if hasattr(output, 'tool_calls'):
            for tool_call in output.tool_calls:
                print(f"Tool: {tool_call.function.name}")
                print(f"Args: {tool_call.function.arguments}")
```

### Performance Benefits for Local Development

1. **Connection Reuse**: SDK maintains persistent connections to Dapr sidecar
2. **Efficient Serialization**: Optimized protobuf/gRPC communication
3. **Reduced Overhead**: No HTTP parsing overhead for local communication
4. **Better Resource Management**: Automatic connection pooling and cleanup

### Migration from HTTP to SDK

If you have existing HTTP-based tests, migration is straightforward:

```python
# Before: Raw HTTP
response = requests.post(
    "http://localhost:3500/v1.0/conversation/anthropic",
    json={"inputs": [{"role": "user", "content": "Hello"}]}
)
result = response.json()["outputs"][0]["result"]

# After: SDK
with DaprClient() as client:
    response = client.invoke_conversation(
        name="anthropic",
        inputs=[{"role": "user", "content": "Hello"}]
    )
    result = response.outputs[0].result
```

This approach provides a much better developer experience for local development scenarios where you have control over all the components.