#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Unit tests for AgentToolExecutor.register_tool()"""

from unittest.mock import patch

import pytest
from dapr_agents.tool.executor import AgentToolExecutor
from dapr_agents.tool.base import AgentTool
from dapr_agents.types import AgentToolExecutorError


class TestAgentToolExecutorRegisterTool:
    """Test suite for AgentToolExecutor.register_tool() method."""

    def test_register_agent_tool_directly(self):
        """Test registering an AgentTool instance directly."""
        executor = AgentToolExecutor()

        tool = AgentTool(name="TestTool", description="A test tool", func=None)

        executor.register_tool(tool)

        # Verify tool was registered
        assert len(executor._tools_map) == 1
        assert executor.get_tool("TestTool") is tool

    def test_register_callable_with_docstring(self):
        """Test registering a callable function with docstring."""
        executor = AgentToolExecutor()

        def my_function(x: int, y: int) -> int:
            """Adds two numbers together."""
            return x + y

        executor.register_tool(my_function)

        assert len(executor._tools_map) == 1
        tool = executor.get_tool("my_function")
        assert tool is not None
        assert tool.name == "my_function"
        assert "Adds two numbers" in tool.description

    def test_register_tool_name_normalization(self):
        """Test that tool lookup uses normalized names (lowercase, no spaces/underscores)."""
        executor = AgentToolExecutor()

        tool = AgentTool(
            name="My_Test_Tool", description="Test normalization", func=None
        )

        executor.register_tool(tool)

        # All these should find the same tool due to normalization
        assert executor.get_tool("My_Test_Tool") is tool
        assert executor.get_tool("my_test_tool") is tool
        assert executor.get_tool("MyTestTool") is tool
        assert executor.get_tool("my test tool") is tool

    def test_register_duplicate_tool_raises(self):
        """Registering a duplicate tool name raises AgentToolExecutorError."""
        executor = AgentToolExecutor()
        tool1 = AgentTool(name="MyTool", description="First tool", func=None)
        tool2 = AgentTool(
            name="MyTool", description="Second tool with same name", func=None
        )
        executor.register_tool(tool1)
        with pytest.raises(AgentToolExecutorError, match="already registered"):
            executor.register_tool(tool2)
        assert executor.get_tool("MyTool") is tool1

    def test_register_duplicate_with_normalized_name_raises(self):
        """Normalized duplicate names (case/spaces/underscores) also raise."""
        executor = AgentToolExecutor()
        tool1 = AgentTool(name="my_tool", description="First", func=None)
        tool2 = AgentTool(name="My Tool", description="Second", func=None)
        executor.register_tool(tool1)
        with pytest.raises(AgentToolExecutorError, match="already registered"):
            executor.register_tool(tool2)
        assert executor.get_tool("My_Tool") is tool1

    def test_register_multiple_distinct_tools(self):
        """Test registering multiple different tools."""
        executor = AgentToolExecutor()

        tool1 = AgentTool(name="Tool1", description="First", func=None)
        tool2 = AgentTool(name="Tool2", description="Second", func=None)
        tool3 = AgentTool(name="Tool3", description="Third", func=None)

        executor.register_tool(tool1)
        executor.register_tool(tool2)
        executor.register_tool(tool3)

        assert len(executor._tools_map) == 3
        assert executor.get_tool("Tool1") is tool1
        assert executor.get_tool("Tool2") is tool2
        assert executor.get_tool("Tool3") is tool3

    def test_register_callable_without_docstring_raises_error(self):
        """Test that callables without docstrings are rejected."""
        executor = AgentToolExecutor()

        def simple_func():
            pass

        # AgentTool.from_func enforces that callables must have docstrings
        with pytest.raises(AgentToolExecutorError):
            executor.register_tool(simple_func)

    def test_register_callable_with_arguments(self):
        """Test registering callable with various argument types."""
        executor = AgentToolExecutor()

        def complex_func(a: str, b: int, c=None):
            """Process data with multiple arguments."""
            return f"{a}:{b}"

        executor.register_tool(complex_func)

        tool = executor.get_tool("complex_func")
        assert tool is not None
        assert tool.args_model is not None

    def test_register_tool_invalid_type_raises_error(self):
        """Test that invalid tool types are rejected."""
        executor = AgentToolExecutor()

        # Try to register something that's neither AgentTool nor Callable
        invalid_tool = "not a tool"

        with pytest.raises(TypeError) as exc_info:
            executor.register_tool(invalid_tool)

        assert "Unsupported tool type" in str(exc_info.value)

    def test_register_callable_that_fails_conversion(self):
        """Test error handling when callable conversion fails."""
        executor = AgentToolExecutor()

        # Create a callable that will fail conversion (missing docstring)
        def bad_tool_func():
            pass

        def mock_from_func(cls, func):
            raise ValueError("Docstring validation failed")

        with patch.object(AgentTool, "from_func", classmethod(mock_from_func)):
            with pytest.raises(AgentToolExecutorError) as exc_info:
                executor.register_tool(bad_tool_func)

            assert "Failed to convert callable" in str(exc_info.value)

    def test_register_tool_with_initialization_list(self):
        """Test that tools provided during init are registered."""
        tool1 = AgentTool(name="InitTool1", description="First", func=None)
        tool2 = AgentTool(name="InitTool2", description="Second", func=None)

        executor = AgentToolExecutor(tools=[tool1, tool2])

        assert len(executor._tools_map) == 2
        assert executor.get_tool("InitTool1") is tool1
        assert executor.get_tool("InitTool2") is tool2

    def test_register_tool_preserves_tool_identity(self):
        """Test that registered tool maintains its identity."""
        executor = AgentToolExecutor()

        tool = AgentTool(
            name="IdentityTest", description="Test identity preservation", func=None
        )

        executor.register_tool(tool)

        # Retrieved tool should be the exact same object
        retrieved = executor.get_tool("IdentityTest")
        assert retrieved is tool
        assert id(retrieved) == id(tool)

    def test_get_tool_not_found_returns_none(self):
        """Test that retrieving non-existent tool returns None."""
        executor = AgentToolExecutor()

        tool = AgentTool(name="OnlyTool", description="Single", func=None)
        executor.register_tool(tool)

        # Try to get a tool that doesn't exist
        result = executor.get_tool("NonExistent")
        assert result is None

    def test_list_tools_returns_all_registered(self):
        """Test that list_tools returns all registered tools."""
        executor = AgentToolExecutor()

        tool1 = AgentTool(name="Tool1", description="First", func=None)
        tool2 = AgentTool(name="Tool2", description="Second", func=None)
        tool3 = AgentTool(name="Tool3", description="Third", func=None)

        executor.register_tool(tool1)
        executor.register_tool(tool2)
        executor.register_tool(tool3)

        tools = executor.list_tools()
        assert len(tools) == 3
        assert tool1 in tools
        assert tool2 in tools
        assert tool3 in tools
