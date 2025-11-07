"""Tests for tool result serialization utilities."""
import json

import pytest
from pydantic import BaseModel

from dapr_agents.tool.utils.serialization import serialize_tool_result


class MockPydanticV2Model(BaseModel):
    """Mock Pydantic v2 model for testing."""

    name: str
    value: int


class MockPydanticV1Model:
    """Mock Pydantic v1-style model for testing."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def dict(self):
        """Pydantic v1-style dict method."""
        return {"name": self.name, "value": self.value}


class MockObjectWithDict:
    """Mock object with __dict__ attribute."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value


class MockNonSerializableObject:
    """Mock object that can't be JSON serialized."""

    def __init__(self, data):
        self._data = data

    def __str__(self):
        return f"MockObject({self._data})"


class TrulyNonSerializableObject:
    """Object that truly cannot be JSON serialized (no __dict__, contains non-serializable data)."""

    __slots__ = ["data"]  # Prevents __dict__ creation

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return f"NonSerializable({self.data})"


class TestSerializeToolResult:
    """Test suite for serialize_tool_result function."""

    def test_serialize_string_returns_as_is(self):
        """Test that string results are returned unchanged."""
        result = "Hello, World!"
        assert serialize_tool_result(result) == "Hello, World!"

    def test_serialize_empty_string(self):
        """Test that empty strings are returned unchanged."""
        result = ""
        assert serialize_tool_result(result) == ""

    def test_serialize_integer(self):
        """Test serialization of integer primitive."""
        result = 42
        assert serialize_tool_result(result) == "42"

    def test_serialize_float(self):
        """Test serialization of float primitive."""
        result = 3.14159
        assert serialize_tool_result(result) == "3.14159"

    def test_serialize_boolean(self):
        """Test serialization of boolean values."""
        assert serialize_tool_result(True) == "true"
        assert serialize_tool_result(False) == "false"

    def test_serialize_none(self):
        """Test serialization of None value."""
        result = None
        assert serialize_tool_result(result) == "null"

    def test_serialize_dict(self):
        """Test serialization of dictionary."""
        result = {"key": "value", "number": 123}
        serialized = serialize_tool_result(result)
        assert json.loads(serialized) == {"key": "value", "number": 123}

    def test_serialize_list_of_primitives(self):
        """Test serialization of list with primitive types."""
        result = [1, 2, 3, "four", 5.0]
        serialized = serialize_tool_result(result)
        assert json.loads(serialized) == [1, 2, 3, "four", 5.0]

    def test_serialize_empty_list(self):
        """Test serialization of empty list."""
        result = []
        assert serialize_tool_result(result) == "[]"

    def test_serialize_nested_dict(self):
        """Test serialization of nested dictionary structures."""
        result = {
            "user": {"name": "John", "age": 30},
            "items": [{"id": 1}, {"id": 2}],
        }
        serialized = serialize_tool_result(result)
        assert json.loads(serialized) == result

    def test_serialize_pydantic_v2_model(self):
        """Test serialization of Pydantic v2 model."""
        model = MockPydanticV2Model(name="test", value=100)
        serialized = serialize_tool_result(model)
        assert json.loads(serialized) == {"name": "test", "value": 100}

    def test_serialize_list_of_pydantic_v2_models(self):
        """Test serialization of list containing Pydantic v2 models."""
        models = [
            MockPydanticV2Model(name="first", value=1),
            MockPydanticV2Model(name="second", value=2),
            MockPydanticV2Model(name="third", value=3),
        ]
        serialized = serialize_tool_result(models)
        expected = [
            {"name": "first", "value": 1},
            {"name": "second", "value": 2},
            {"name": "third", "value": 3},
        ]
        assert json.loads(serialized) == expected

    def test_serialize_pydantic_v1_model(self):
        """Test serialization of Pydantic v1-style model with dict() method."""
        model = MockPydanticV1Model(name="legacy", value=42)
        serialized = serialize_tool_result(model)
        assert json.loads(serialized) == {"name": "legacy", "value": 42}

    def test_serialize_list_of_pydantic_v1_models(self):
        """Test serialization of list containing Pydantic v1-style models."""
        models = [
            MockPydanticV1Model(name="old1", value=10),
            MockPydanticV1Model(name="old2", value=20),
        ]
        serialized = serialize_tool_result(models)
        expected = [
            {"name": "old1", "value": 10},
            {"name": "old2", "value": 20},
        ]
        assert json.loads(serialized) == expected

    def test_serialize_object_with_dict(self):
        """Test serialization of object with __dict__ attribute."""
        obj = MockObjectWithDict(name="custom", value=999)
        serialized = serialize_tool_result(obj)
        assert json.loads(serialized) == {"name": "custom", "value": 999}

    def test_serialize_list_of_objects_with_dict(self):
        """Test serialization of list containing objects with __dict__."""
        objects = [
            MockObjectWithDict(name="obj1", value=1),
            MockObjectWithDict(name="obj2", value=2),
        ]
        serialized = serialize_tool_result(objects)
        expected = [
            {"name": "obj1", "value": 1},
            {"name": "obj2", "value": 2},
        ]
        assert json.loads(serialized) == expected

    def test_serialize_mixed_list(self):
        """Test serialization of list with mixed types."""
        models = [
            MockPydanticV2Model(name="pydantic", value=1),
            MockObjectWithDict(name="custom", value=2),
            {"dict": "item"},
            "string_item",
            123,
        ]
        serialized = serialize_tool_result(models)
        expected = [
            {"name": "pydantic", "value": 1},
            {"name": "custom", "value": 2},
            {"dict": "item"},
            "string_item",
            123,
        ]
        assert json.loads(serialized) == expected

    def test_serialize_object_with_dict_attribute(self):
        """Test that objects with __dict__ are serialized via their attributes."""
        obj = MockNonSerializableObject("special_data")
        serialized = serialize_tool_result(obj)
        # Objects with __dict__ are serialized via their attributes
        assert json.loads(serialized) == {"_data": "special_data"}

    def test_serialize_truly_non_serializable_object_fallback(self):
        """Test that truly non-serializable objects fall back to string representation."""
        obj = TrulyNonSerializableObject("test_data")
        serialized = serialize_tool_result(obj)
        # Should fall back to str() representation
        assert serialized == "NonSerializable(test_data)"

    def test_serialize_complex_nested_structure(self):
        """Test serialization of complex nested data structure."""
        result = {
            "status": "success",
            "data": {
                "users": [
                    {"id": 1, "name": "Alice"},
                    {"id": 2, "name": "Bob"},
                ],
                "metadata": {"count": 2, "page": 1},
            },
        }
        serialized = serialize_tool_result(result)
        assert json.loads(serialized) == result

    def test_serialize_list_preserves_order(self):
        """Test that list order is preserved during serialization."""
        models = [MockPydanticV2Model(name=f"item_{i}", value=i) for i in range(10)]
        serialized = serialize_tool_result(models)
        deserialized = json.loads(serialized)
        for i, item in enumerate(deserialized):
            assert item == {"name": f"item_{i}", "value": i}

    def test_serialize_unicode_strings(self):
        """Test serialization of unicode strings."""
        result = "Hello 世界 🌍"
        assert serialize_tool_result(result) == "Hello 世界 🌍"

    def test_serialize_special_characters(self):
        """Test serialization of strings with special characters."""
        result = {"message": 'He said, "Hello!"'}
        serialized = serialize_tool_result(result)
        assert json.loads(serialized) == result

    def test_serialize_numeric_edge_cases(self):
        """Test serialization of numeric edge cases."""
        result = {"zero": 0, "negative": -42, "large": 1e10}
        serialized = serialize_tool_result(result)
        deserialized = json.loads(serialized)
        assert deserialized["zero"] == 0
        assert deserialized["negative"] == -42
        assert deserialized["large"] == 1e10

    def test_serialize_empty_dict(self):
        """Test serialization of empty dictionary."""
        result = {}
        assert serialize_tool_result(result) == "{}"

    def test_serialize_flight_example_from_docstring(self):
        """Test the exact example from the function's docstring."""

        class Flight(BaseModel):
            airline: str
            price: float

        flights = [Flight(airline="SkyHigh", price=450.0)]
        serialized = serialize_tool_result(flights)
        assert json.loads(serialized) == [{"airline": "SkyHigh", "price": 450.0}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
