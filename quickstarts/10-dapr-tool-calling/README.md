# Quickstart 10: Dapr Tool Calling

This quickstart demonstrates **tool calling** with Dapr conversation components using three different approaches, from simple to advanced.

## 🎯 What You'll Learn

- **Simple Agent tool calling** - Clean, straightforward approach
- **Enhanced DaprChatClient** - Low-level control with multi-turn conversations
- **AssistantAgent workflows** - High-level persistent agents with automatic tool iteration

## 🚀 Quick Start

### Prerequisites
- **Dapr CLI** installed and initialized (`dapr init`)
- **Python 3.10+** 
- **API keys** for real LLM providers (optional - works with echo for testing)

### 1. Setup Environment
```bash
# Clone and navigate to quickstart
cd quickstarts/10-dapr-tool-calling

# Install dependencies
pip install -r requirements.txt

# Optional: Create .env file with API keys
cp ../../.env.example .env  # Edit with your keys
```

### 2. Start Dapr Infrastructure
```bash
# Start Dapr with components
python ../../tools/run_dapr_dev.py --app-id tool-calling --components ./components
```

### 3. Choose Your Approach

## 📖 Three Approaches to Tool Calling

### 🟢 **Approach 1: Simple Agent** (Recommended for most use cases)

**File**: `simple_agent_example.py`

The cleanest approach - similar to quickstart 03 but with tool calling enabled.

```python
# Configure enhanced DaprChatClient
llm_client = DaprChatClient(component_name="openai")

# Create agent with tools
agent = Agent(
    name="WeatherBot",
    tools=[get_weather, calculate],
    llm=llm_client
)

# Simple usage
await agent.run("What's the weather in Tokyo and calculate 15 * 8?")
```

**Features:**
- ✅ **Simple setup** - Just like quickstart 03
- ✅ **Automatic tool calling** - Agent handles everything
- ✅ **Multiple providers** - OpenAI, Anthropic, Echo
- ✅ **Multiple tools** - Function and class-based tools

**Run it:**
```bash
python simple_agent_example.py --provider openai
python simple_agent_example.py --provider anthropic
python simple_agent_example.py --provider echo  # No API key needed
```

---

### 🟡 **Approach 2: Enhanced DaprChatClient** (For advanced control)

**File**: `enhanced_dapr_client_example.py`

Low-level approach with full control over conversation flow and multi-turn tool calling.

```python
# Enhanced client with tool calling support
client = DaprChatClient(component_name="openai")

# Multi-turn conversation with tools
response = client.generate(messages=messages, tools=tools)
# Handle tool calls manually
# Execute tools and continue conversation
```

**Features:**
- ✅ **Full control** - Manual conversation management
- ✅ **Multi-turn workflows** - Complex tool calling sequences
- ✅ **Raw response access** - Complete conversation building
- ✅ **Streaming support** - Real-time tool calling

**Run it:**
```bash
python enhanced_dapr_client_example.py --provider openai
```

---

### 🔵 **Approach 3: AssistantAgent** (For production workflows)

**File**: `assistant_agent_example.py`

Workflow-based approach with persistent memory, automatic iteration, and advanced state management.

```python
# Workflow-based assistant with persistent state
assistant = AssistantAgent(
    name="ToolAssistant",
    tools=tools,
    llm=DaprChatClient(component_name="openai"),
    memory=ConversationDaprStateMemory(session_id="demo"),
    max_iterations=5
)

# Runs as a persistent service
await assistant.start()
```

**Features:**
- ✅ **Persistent memory** - Conversation history across sessions
- ✅ **Automatic iteration** - Multi-step tool calling workflows
- ✅ **Workflow execution** - Durable, resumable processes
- ✅ **Multi-agent support** - Service-to-service communication
- ✅ **State management** - Built-in persistence via Dapr

**Note:** AssistantAgent runs as a service and requires triggers to execute. Use `trigger_assistant.py` to interact with the running assistant via HTTP calls or workflow triggers.

**Run it:**
```bash
python assistant_agent_example.py --provider openai
python assistant_agent_example.py --provider openai --class-tools  # Use class-based tools
python assistant_agent_example.py --demo  # Interactive demo mode
```

## 🔧 Configuration

### Component Names
All examples support multiple Dapr conversation components:

- **`echo`** - Testing/development (no API key needed)
- **`openai`** - OpenAI GPT models
- **`anthropic`** - Anthropic Claude models
- **`gemini`** - Google Gemini models

### Environment Variables
```bash
# Optional: Set default component
export DAPR_LLM_COMPONENT_DEFAULT=openai

# API Keys (add to .env file)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## 🛠️ Available Tools

All examples include these tools for demonstration:

### Function-based Tools
- **`get_weather(location)`** - Get weather information
- **`calculate(expression)`** - Perform mathematical calculations  
- **`get_time_zone(location)`** - Get timezone information

### Class-based Tools
- **`CurrencyConverter`** - Convert between currencies
- **`TaskManager`** - Manage a simple task list

## 🎯 When to Use Each Approach

| Use Case | Recommended Approach |
|----------|---------------------|
| **Simple tool calling** | 🟢 Simple Agent |
| **Quick prototyping** | �� Simple Agent |
| **Complex multi-turn logic** | 🟡 Enhanced DaprChatClient |
| **Custom conversation flow** | 🟡 Enhanced DaprChatClient |
| **Production workflows** | 🔵 AssistantAgent |
| **Persistent agents** | 🔵 AssistantAgent |
| **Multi-agent systems** | 🔵 AssistantAgent |

## 🔍 Key Differences

| Feature | Simple Agent | Enhanced Client | AssistantAgent |
|---------|-------------|----------------|----------------|
| **Setup Complexity** | Low | Medium | High |
| **Tool Calling** | Automatic | Manual | Automatic |
| **Memory** | None | Manual | Persistent |
| **Multi-turn** | Basic | Advanced | Automatic |
| **State Management** | None | Manual | Built-in |
| **Workflows** | No | No | Yes |
| **Production Ready** | Basic | Advanced | Full |

## 🚀 Next Steps

1. **Start with Simple Agent** - Try `simple_agent_example.py` first
2. **Explore Enhanced Client** - For advanced control needs
3. **Try AssistantAgent** - For production workflow requirements
4. **Build your own tools** - Create custom tools for your use case
5. **Integrate with workflows** - Combine with quickstart 04 (agentic workflows)

## 📚 Related Quickstarts

- **[Quickstart 03](../03-agent-tool-call/)** - Basic agent tool calling
- **[Quickstart 04](../04-agentic-workflow/)** - Agentic workflows
- **[Quickstart 09](../09-dapr-streaming/)** - Streaming conversations

## 🆘 Troubleshooting

### Common Issues

**"Actor runtime disabled"**
```bash
# Restart Dapr with placement service
python ../../tools/run_dapr_dev.py --app-id tool-calling --components ./components
```

**"Component not found"**
```bash
# Check component configuration
ls components/
# Verify component name matches provider argument
```

**Tool calls not working**
```bash
# Test with echo provider first
python simple_agent_example.py --provider echo

# Check API keys for real providers
cat .env
```

---

Ready to explore advanced tool calling with Dapr! 🚀
