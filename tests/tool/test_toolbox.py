"""Tests for AgentTool.from_toolbox and from_toolbox_many methods."""

from typing import Any, Optional
from unittest.mock import MagicMock, patch
import pytest

from pydantic import BaseModel
from mcp.types import CallToolResult, TextContent

from dapr_agents.tool.base import AgentTool


class MockParameterSchema:
    """Mock ParameterSchema from toolbox_core."""

    def __init__(self, name: str, required: bool = True, annotation: type = str):
        self.name = name
        self.required = required
        self.annotation = annotation


class MockToolboxSyncTool:
    """Mock ToolboxSyncTool for testing."""

    def __init__(
        self,
        name: str,
        description: str,
        params: list = None,
        return_value: Any = "success",
        raise_exception: Exception = None,
    ):
        self._name = name
        self._description = description
        self._params = params or []
        self._return_value = return_value
        self._raise_exception = raise_exception

    def __call__(self, **kwargs) -> Any:
        if self._raise_exception:
            raise self._raise_exception
        return self._return_value


class TestFromToolbox:
    """Test suite for AgentTool.from_toolbox class method."""

    def test_basic_tool_conversion(self):
        """Test basic conversion of a ToolboxSyncTool to AgentTool."""
        mock_tool = MockToolboxSyncTool(
            name="get_user",
            description="Fetches a user by ID",
            params=[MockParameterSchema("user_id", required=True, annotation=str)],
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)

        assert agent_tool.name == "GetUser"  # Normalized name
        assert agent_tool.description == "Fetches a user by ID"
        assert agent_tool.args_model is not None
        assert "user_id" in agent_tool.args_model.model_fields

    def test_tool_with_multiple_params(self):
        """Test conversion with multiple parameters including optional ones."""
        mock_tool = MockToolboxSyncTool(
            name="search_orders",
            description="Search orders by criteria",
            params=[
                MockParameterSchema("customer_id", required=True, annotation=str),
                MockParameterSchema("status", required=False, annotation=str),
                MockParameterSchema("limit", required=False, annotation=int),
            ],
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)

        assert agent_tool.args_model is not None
        fields = agent_tool.args_model.model_fields
        assert "customer_id" in fields
        assert "status" in fields
        assert "limit" in fields
        # Required field should not have a default
        assert fields["customer_id"].is_required()
        # Optional fields should have None default
        assert not fields["status"].is_required()
        assert not fields["limit"].is_required()

    def test_tool_with_no_params(self):
        """Test conversion of a tool with no parameters."""
        mock_tool = MockToolboxSyncTool(
            name="get_current_time",
            description="Returns the current server time",
            params=[],
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)

        assert agent_tool.name == "GetCurrentTime"
        assert agent_tool.description == "Returns the current server time"
        # Should have an empty args model, not None
        assert agent_tool.args_model is not None
        assert len(agent_tool.args_model.model_fields) == 0

    def test_tool_with_none_params(self):
        """Test conversion when params is None."""
        mock_tool = MockToolboxSyncTool(
            name="ping",
            description="Health check endpoint",
            params=None,
        )
        # Manually set to None to simulate edge case
        mock_tool._params = None

        agent_tool = AgentTool.from_toolbox(mock_tool)

        assert agent_tool.name == "Ping"
        # Should have an empty args model even when params is None
        assert agent_tool.args_model is not None

    def test_tool_execution_success(self):
        """Test that the wrapped tool executes correctly."""
        mock_tool = MockToolboxSyncTool(
            name="add_numbers",
            description="Adds two numbers",
            params=[
                MockParameterSchema("a", required=True, annotation=int),
                MockParameterSchema("b", required=True, annotation=int),
            ],
            return_value=42,
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)
        result = agent_tool.func(a=10, b=32)

        assert isinstance(result, CallToolResult)
        assert result.isError is False
        assert len(result.content) == 1
        assert result.content[0].text == "42"

    def test_tool_execution_with_complex_result(self):
        """Test that complex return values are stringified."""
        mock_tool = MockToolboxSyncTool(
            name="get_order",
            description="Gets order details",
            params=[MockParameterSchema("order_id", required=True, annotation=str)],
            return_value={"id": "ORD-001", "status": "shipped", "items": 3},
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)
        result = agent_tool.func(order_id="ORD-001")

        assert isinstance(result, CallToolResult)
        assert result.isError is False
        assert "ORD-001" in result.content[0].text
        assert "shipped" in result.content[0].text

    def test_tool_execution_error_handling(self):
        """Test that exceptions are properly wrapped in CallToolResult."""
        mock_tool = MockToolboxSyncTool(
            name="failing_tool",
            description="A tool that always fails",
            params=[],
            raise_exception=ValueError("Something went wrong"),
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)
        result = agent_tool.func()

        assert isinstance(result, CallToolResult)
        assert result.isError is True
        assert "ValueError" in result.content[0].text
        assert "Something went wrong" in result.content[0].text

    def test_tool_execution_validation_error(self):
        """Test handling of Pydantic ValidationError."""
        from pydantic import ValidationError

        # Create a mock that raises ValidationError
        mock_tool = MockToolboxSyncTool(
            name="validate_tool",
            description="A tool with validation",
            params=[MockParameterSchema("value", required=True, annotation=int)],
        )
        mock_tool._raise_exception = ValidationError.from_exception_data(
            "validation error",
            [
                {
                    "type": "int_parsing",
                    "loc": ("value",),
                    "msg": "Input should be a valid integer",
                    "input": "not_an_int",
                }
            ],
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)
        result = agent_tool.func(value="not_an_int")

        assert isinstance(result, CallToolResult)
        assert result.isError is True
        assert "ValidationError" in result.content[0].text

    def test_name_normalization(self):
        """Test that tool names are properly normalized."""
        test_cases = [
            ("get_user_by_id", "GetUserById"),
            ("simple", "Simple"),
            ("UPPERCASE", "Uppercase"),
            ("mixed_Case_name", "MixedCaseName"),
        ]

        for original, expected in test_cases:
            mock_tool = MockToolboxSyncTool(
                name=original,
                description="Test tool",
                params=[],
            )
            agent_tool = AgentTool.from_toolbox(mock_tool)
            assert (
                agent_tool.name == expected
            ), f"Expected {expected}, got {agent_tool.name}"


class TestFromToolboxMany:
    """Test suite for AgentTool.from_toolbox_many class method."""

    def test_batch_conversion(self):
        """Test batch conversion of multiple ToolboxSyncTools."""
        mock_tools = [
            MockToolboxSyncTool(
                name="tool_one",
                description="First tool",
                params=[MockParameterSchema("param1", required=True)],
            ),
            MockToolboxSyncTool(
                name="tool_two",
                description="Second tool",
                params=[],
            ),
            MockToolboxSyncTool(
                name="tool_three",
                description="Third tool",
                params=[
                    MockParameterSchema("a", required=True),
                    MockParameterSchema("b", required=False),
                ],
            ),
        ]

        agent_tools = AgentTool.from_toolbox_many(mock_tools)

        assert len(agent_tools) == 3
        assert all(isinstance(t, AgentTool) for t in agent_tools)
        assert agent_tools[0].name == "ToolOne"
        assert agent_tools[1].name == "ToolTwo"
        assert agent_tools[2].name == "ToolThree"

    def test_empty_list(self):
        """Test conversion of empty list."""
        agent_tools = AgentTool.from_toolbox_many([])
        assert agent_tools == []

    def test_single_tool_list(self):
        """Test conversion of single-item list."""
        mock_tools = [
            MockToolboxSyncTool(
                name="only_tool",
                description="The only tool",
                params=[],
            )
        ]

        agent_tools = AgentTool.from_toolbox_many(mock_tools)

        assert len(agent_tools) == 1
        assert agent_tools[0].name == "OnlyTool"


class TestFromToolboxIntegration:
    """Integration-style tests for the from_toolbox workflow."""

    def test_tool_can_be_invoked_with_validated_args(self):
        """Test that the args_model correctly validates input."""
        mock_tool = MockToolboxSyncTool(
            name="create_task",
            description="Creates a new task",
            params=[
                MockParameterSchema("title", required=True, annotation=str),
                MockParameterSchema("priority", required=False, annotation=int),
            ],
            return_value="Task created",
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)

        # Validate args using the generated model
        validated_args = agent_tool.args_model(title="My Task", priority=5)
        assert validated_args.title == "My Task"
        assert validated_args.priority == 5

        # Execute with validated args
        result = agent_tool.func(**validated_args.model_dump())
        assert isinstance(result, CallToolResult)
        assert result.isError is False

    def test_tool_with_optional_args_uses_defaults(self):
        """Test that optional parameters work correctly."""
        mock_tool = MockToolboxSyncTool(
            name="search",
            description="Search with optional filters",
            params=[
                MockParameterSchema("query", required=True, annotation=str),
                MockParameterSchema("limit", required=False, annotation=int),
            ],
            return_value="Results found",
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)

        # Only provide required args
        validated_args = agent_tool.args_model(query="test")
        assert validated_args.query == "test"
        assert validated_args.limit is None

    def test_tool_args_model_validation_failure(self):
        """Test that invalid args are rejected by the args model."""
        mock_tool = MockToolboxSyncTool(
            name="typed_tool",
            description="Tool with typed params",
            params=[
                MockParameterSchema("count", required=True, annotation=int),
            ],
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)

        # The args_model should reject missing required fields
        with pytest.raises(Exception):  # Pydantic ValidationError
            agent_tool.args_model()  # Missing required 'count'


class TestFromToolboxEdgeCases:
    """Edge case tests for from_toolbox."""

    def test_param_without_annotation(self):
        """Test handling of param without explicit annotation."""

        class ParamWithoutAnnotation:
            def __init__(self, name: str, required: bool = True):
                self.name = name
                self.required = required
                # No annotation attribute

        mock_tool = MockToolboxSyncTool(
            name="legacy_tool",
            description="A legacy tool",
            params=[ParamWithoutAnnotation("old_param")],
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)

        # Should still work, defaulting annotation to Any
        assert agent_tool.args_model is not None
        assert "old_param" in agent_tool.args_model.model_fields

    def test_special_characters_in_description(self):
        """Test tool with special characters in description."""
        mock_tool = MockToolboxSyncTool(
            name="special_tool",
            description='Tool with "quotes" and <html> & symbols',
            params=[],
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)

        assert 'Tool with "quotes"' in agent_tool.description
        assert "<html>" in agent_tool.description

    def test_unicode_in_tool_name_and_description(self):
        """Test tool with unicode characters."""
        mock_tool = MockToolboxSyncTool(
            name="café_finder",
            description="Finds nearby cafés ☕",
            params=[MockParameterSchema("city", required=True, annotation=str)],
        )

        agent_tool = AgentTool.from_toolbox(mock_tool)

        assert "☕" in agent_tool.description
