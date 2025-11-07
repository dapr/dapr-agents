import pytest
from unittest.mock import AsyncMock, Mock
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.schemas import AgentWorkflowEntry, AgentWorkflowState
from dapr_agents.tool.base import AgentTool


@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
    monkeypatch.setattr(DurableAgent, "save_state", lambda self: None)

    # The following monkeypatches are for legacy compatibility with dict-like access in tests.
    # If AgentWorkflowState supports dict-like access natively, these can be removed.
    def _getitem(self, key):
        return getattr(self, key)

    def _setdefault(self, key, default):
        if hasattr(self, key):
            return getattr(self, key)
        setattr(self, key, default)
        return default

    AgentWorkflowState.__getitem__ = _getitem
    AgentWorkflowState.setdefault = _setdefault
    # Patch DaprStateStore to use a mock DaprClient that supports context manager
    import dapr_agents.storage.daprstores.statestore as statestore

    class MockDaprClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def save_state(self, *args, **kwargs):
            pass

        def get_state(self, *args, **kwargs):
            class R:
                data = "{}"
                etag = "etag"

            return R()

        def execute_state_transaction(self, *args, **kwargs):
            pass

    statestore.DaprClient = MockDaprClient

    # Patch out agent registration logic (skip state store entirely)
    def mock_register_agentic_system(self):
        pass

    monkeypatch.setattr(
        DurableAgent, "register_agentic_system", mock_register_agentic_system
    )

    yield


@pytest.fixture
def mock_mcp_tool():
    mcp_tool = Mock()
    mcp_tool.name = "add"
    mcp_tool.description = "Add two numbers"
    # Provide an input schema so the tool expects 'a' and 'b' as direct arguments
    mcp_tool.inputSchema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }
    return mcp_tool


@pytest.fixture
def mock_mcp_session():
    # Simulate a streamable HTTP response by returning the sum as a string
    import json

    async def fake_call_tool(*args, **kwargs):
        a = b = 0
        # Handle all possible argument patterns for tool execution
        if len(args) >= 2:
            if isinstance(args[1], dict):
                a = int(args[1].get("a", 0))
                b = int(args[1].get("b", 0))
            elif isinstance(args[1], str):
                try:
                    data = json.loads(args[1])
                    if isinstance(data, dict):
                        a = int(data.get("a", 0))
                        b = int(data.get("b", 0))
                except Exception:
                    a = b = 0
        elif "a" in kwargs and "b" in kwargs:
            try:
                a = int(kwargs["a"])
                b = int(kwargs["b"])
            except Exception:
                a = b = 0
        elif args and isinstance(args[0], dict):
            try:
                a = int(args[0].get("a", 0))
                b = int(args[0].get("b", 0))
            except Exception:
                a = b = 0
        elif args and isinstance(args[0], str):
            try:
                data = json.loads(args[0])
                if isinstance(data, dict):
                    a = int(data.get("a", 0))
                    b = int(data.get("b", 0))
            except Exception:
                a = b = 0
        return str(a + b)

    session = Mock()
    session.call_tool = AsyncMock(side_effect=fake_call_tool)
    return session


@pytest.fixture
def durable_agent_with_mcp_tool(mock_mcp_tool, mock_mcp_session):
    from dapr_agents.agents.configs import AgentPubSubConfig, AgentStateConfig
    from dapr_agents.storage.daprstores.stateservice import StateStoreService

    agent_tool = AgentTool.from_mcp(mock_mcp_tool, session=mock_mcp_session)

    agent = DurableAgent(
        name="TestDurableAgent",
        role="Math Assistant",
        goal="Help humans do math",
        instructions=["Test math instructions"],
        tools=[agent_tool],
        pubsub=AgentPubSubConfig(
            pubsub_name="testpubsub",
        ),
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
    )
    return agent


def test_execute_tool_activity_with_mcp_tool(durable_agent_with_mcp_tool):
    # Test the mocked MCP tool (add) with DurableAgent
    instance_id = "test-instance-123"

    # Use AgentWorkflowEntry for state setup
    entry = AgentWorkflowEntry(
        input_value="What is 2 plus 2?",
        source=None,
        triggering_workflow_instance_id=None,
        workflow_instance_id=instance_id,
        workflow_name="AgenticWorkflow",
        status="RUNNING",
        messages=[],
        tool_history=[],
    )
    durable_agent_with_mcp_tool._state_model.instances[instance_id] = entry

    # Print available tool names for debugging
    tool_names = [t.name for t in durable_agent_with_mcp_tool.tool_executor.tools]
    print("Available tool names (unit test):", tool_names)
    # Use the correct tool name as present in the executor
    tool_name = next(
        (n for n in tool_names if n.lower().startswith("add")), tool_names[0]
    )

    # Create mock context
    mock_ctx = Mock()

    # Call run_tool activity with new signature (ctx, payload)
    result = durable_agent_with_mcp_tool.run_tool(
        mock_ctx,
        {
            "instance_id": instance_id,
            "tool_call": {
                "id": "call_123",
                "type": "function",
                "function": {"name": tool_name, "arguments": '{"a": 2, "b": 2}'},
            },
        },
    )

    # Verify the tool result structure
    assert result["tool_call_id"] == "call_123"
    assert result["tool_name"] == tool_name
    assert result["execution_result"] == "4"  # Serialized as string


# Shared fixture to start the math server with streamable HTTP
@pytest.fixture(scope="module")
def start_math_server_http():
    import subprocess
    import time

    proc = subprocess.Popen(
        [
            "python",
            "tests/agents/durableagent/test_mcp_math_server.py",
            "--server_type",
            "streamable-http",
            "--port",
            "8000",
        ]
    )
    time.sleep(1.5)  # Give the server time to start
    yield
    proc.terminate()
    proc.wait()


# Helper to get agent tools from a real MCP server
async def get_agent_tools_from_http():
    from dapr_agents.tool.mcp import MCPClient

    client = MCPClient()
    await client.connect_streamable_http(
        server_name="local", url="http://localhost:8000/mcp/"
    )
    return client.get_all_tools()


@pytest.mark.asyncio
async def test_add_tool_with_real_server_http(start_math_server_http):
    from dapr_agents import Agent

    agent_tools = await get_agent_tools_from_http()
    agent = Agent(name="MathAgent", role="Math Assistant", tools=agent_tools)
    # Print available tool names for debugging
    tool_names = [t.name for t in agent_tools]
    print("Available tool names:", tool_names)
    # Use the correct tool name as provided by the MCP server
    tool_name = next(
        (n for n in tool_names if n.lower().startswith("add")), tool_names[0]
    )
    result = await agent.tool_executor.run_tool(tool_name, a=2, b=2)
    assert result == "4"


@pytest.mark.asyncio
async def test_durable_agent_with_real_server_http(start_math_server_http):
    agent_tools = await get_agent_tools_from_http()
    from dapr_agents.agents.configs import AgentPubSubConfig, AgentStateConfig
    from dapr_agents.storage.daprstores.stateservice import StateStoreService

    agent = DurableAgent(
        name="TestDurableAgent",
        role="Math Assistant",
        goal="Help humans do math",
        instructions=["Test math instructions"],
        tools=agent_tools,
        pubsub=AgentPubSubConfig(
            pubsub_name="testpubsub",
        ),
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
    )

    instance_id = "test-instance-456"
    # Use AgentWorkflowEntry for state setup
    entry = AgentWorkflowEntry(
        input_value="What is 2 plus 2?",
        source=None,
        triggering_workflow_instance_id=None,
        workflow_instance_id=instance_id,
        workflow_name="AgenticWorkflow",
        status="RUNNING",
        messages=[],
        tool_history=[],
    )
    agent._state_model.instances[instance_id] = entry

    # Print available tool names
    tool_names = [t.name for t in agent.tool_executor.tools]
    print("Available tool names (integration test):", tool_names)

    tool_name = next(
        (n for n in tool_names if n.lower().startswith("add")), tool_names[0]
    )

    #  Create mock context
    mock_ctx = Mock()

    # Call run_tool activity with new signature (ctx, payload)
    # Note: run_tool is synchronous but uses _run_asyncio_task internally,
    # so when called from an async test context, we need to run it in a thread
    import asyncio

    result = await asyncio.to_thread(
        agent.run_tool,
        mock_ctx,
        {
            "instance_id": instance_id,
            "tool_call": {
                "id": "call_456",
                "type": "function",
                "function": {"name": tool_name, "arguments": '{"a": 2, "b": 2}'},
            },
        },
    )

    # Verify the tool result structure
    assert result["tool_call_id"] == "call_456"
    assert result["tool_name"] == tool_name
    assert result["execution_result"] == "4"  # Serialized as string
