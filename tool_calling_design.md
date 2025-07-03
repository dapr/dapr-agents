# Tool Calling Design: Dapr Python SDK ↔ dapr-agents

This document outlines the exact JSON format and interface design for tool calling between the Dapr Python SDK and dapr-agents.

## Current ConversationInput Structure

From the Dapr Python SDK, the current `ConversationInput` structure is:

```python
@dataclass
class ConversationInput:
    """A single input message for the conversation."""
    
    content: str
    role: Optional[str] = None
    scrub_pii: Optional[bool] = None
```

## Proposed Extended ConversationInput for Tool Calling

To support tool calling, we need to extend the `ConversationInput` structure:

```python
@dataclass
class ConversationInput:
    """A single input message for the conversation."""
    
    content: str
    role: Optional[str] = None
    scrub_pii: Optional[bool] = None
    
    # NEW: Tool calling support
    tools: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool response messages
```

## Tool Definition Format

Tools should be provided in OpenAI-compatible format that dapr-agents already supports:

### Single Tool Definition
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather conditions for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "The city and state/country, e.g. 'San Francisco, CA'"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature unit",
          "default": "fahrenheit"
        }
      },
      "required": ["location"]
    }
  }
}
```

### Multiple Tools Array
```json
[
  {
    "type": "function", 
    "function": {
      "name": "get_weather",
      "description": "Get current weather conditions for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state/country"
          }
        },
        "required": ["location"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "calculate_distance", 
      "description": "Calculate distance between two locations",
      "parameters": {
        "type": "object",
        "properties": {
          "origin": {"type": "string"},
          "destination": {"type": "string"}
        },
        "required": ["origin", "destination"]
      }
    }
  }
]
```

## Complete Request/Response Flow

### 1. Initial Request with Tools (SDK → dapr-agents)

```python
from dapr.clients.grpc._request import ConversationInput

# Request with tools available
inputs = [
    ConversationInput(
        content="What's the weather like in San Francisco?",
        role="user",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather conditions for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state/country"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    )
]
```

### 2. Assistant Response with Tool Call (dapr-agents → SDK)

When the LLM wants to call a tool, the response will contain:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"San Francisco, CA\"}"
      }
    }
  ]
}
```

### 3. Tool Execution Result (SDK → dapr-agents)

The SDK should execute the tool and send back the result:

```python
inputs = [
    ConversationInput(
        content='{"temperature": 72, "condition": "sunny", "humidity": 65}',
        role="tool",
        tool_call_id="call_abc123",
        name="get_weather"
    )
]
```

### 4. Final Assistant Response (dapr-agents → SDK)

```json
{
  "role": "assistant", 
  "content": "The weather in San Francisco is currently sunny with a temperature of 72°F and 65% humidity."
}
```

## Implementation Requirements

### Dapr Python SDK Changes

1. **Extend ConversationInput**: Add `tools`, `tool_call_id`, and `name` fields
2. **Tool Call Detection**: Parse response to detect tool calls
3. **Tool Execution Framework**: Provide utilities to register and execute tools
4. **Response Handling**: Handle tool call responses properly

### dapr-agents Changes (Current Implementation)

dapr-agents already supports tool calling through:

1. **Tool Processing**: `RequestHandler.process_params()` handles tools
2. **Tool Formatting**: `ToolHelper.format_tool()` formats tools to OpenAI format
3. **Response Processing**: Stream handler processes tool call responses

The current `process_params` method already handles tools:

```python
@staticmethod
def process_params(
    params: Dict[str, Any],
    llm_provider: str,
    tools: Optional[List[Dict[str, Any]]] = None,  # ← Already supported
    response_format: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
    structured_mode: Literal["json", "function_call"] = "json",
) -> Dict[str, Any]:
    if tools:
        logger.info("Tools are available in the request.")
        params["tools"] = [
            ToolHelper.format_tool(tool, tool_format=llm_provider) for tool in tools
        ]
    # ...
```

## Complete Example Usage

```python
from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput

# Define tools
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather", 
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}

# Initial request
with DaprClient() as client:
    # 1. Send user message with available tools
    response = client.converseWithCloud(
        component='openai',
        inputs=[ConversationInput(
            content="What's the weather in NYC?",
            role="user",
            tools=[weather_tool]
        )]
    )
    
    # 2. Check if LLM wants to call a tool
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        
        # 3. Execute the tool (user's responsibility)
        if tool_call.function.name == "get_weather":
            weather_data = get_weather(tool_call.function.arguments)
            
            # 4. Send tool result back
            final_response = client.converseWithCloud(
                component='openai',
                inputs=[ConversationInput(
                    content=json.dumps(weather_data),
                    role="tool",
                    tool_call_id=tool_call.id,
                    name="get_weather"
                )]
            )
            
            print(final_response.content)
```

## JSON Message Examples

### Tool Call Request Message
```json
{
  "role": "user",
  "content": "What's the weather like?",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather info",
        "parameters": {
          "type": "object", 
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

### Tool Call Response Message  
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function", 
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"New York\"}"
      }
    }
  ]
}
```

### Tool Result Message
```json
{
  "role": "tool",
  "content": "{\"temp\": 75, \"condition\": \"sunny\"}",
  "tool_call_id": "call_123",
  "name": "get_weather"
}
```

### Final Response Message
```json
{
  "role": "assistant",
  "content": "The weather in New York is sunny with a temperature of 75°F."
}
```

## Key Design Decisions

1. **OpenAI Compatibility**: Use OpenAI's tool calling format for maximum compatibility
2. **Backwards Compatibility**: Existing ConversationInput usage remains unchanged  
3. **Minimal SDK Changes**: Only extend ConversationInput, don't break existing APIs
4. **Tool Execution**: Keep tool execution on SDK side for flexibility and security
5. **Streaming Support**: Design works with both streaming and non-streaming responses 