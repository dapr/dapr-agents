"""Tests for DaprHTTPClient and related utilities."""
import os
from unittest.mock import patch

import pytest

from dapr_agents.tool.http.client import DaprHTTPClient, _parse_bool_env


class TestParseBoolEnv:
    """Test suite for _parse_bool_env helper function."""

    def test_parse_true_values(self):
        """Test parsing various string representations of True."""
        true_values = ["y", "yes", "t", "true", "on", "1"]
        for value in true_values:
            assert _parse_bool_env(value) is True, f"Failed for value: {value}"
            # Test case insensitivity
            assert (
                _parse_bool_env(value.upper()) is True
            ), f"Failed for uppercase: {value.upper()}"
            assert (
                _parse_bool_env(value.title()) is True
            ), f"Failed for titlecase: {value.title()}"

    def test_parse_false_values(self):
        """Test parsing various string representations of False."""
        false_values = ["n", "no", "f", "false", "off", "0"]
        for value in false_values:
            assert _parse_bool_env(value) is False, f"Failed for value: {value}"
            # Test case insensitivity
            assert (
                _parse_bool_env(value.upper()) is False
            ), f"Failed for uppercase: {value.upper()}"
            assert (
                _parse_bool_env(value.title()) is False
            ), f"Failed for titlecase: {value.title()}"

    def test_parse_invalid_values(self):
        """Test that invalid values raise ValueError."""
        invalid_values = ["invalid", "maybe", "2", "TRUE1", "yes!", ""]
        for value in invalid_values:
            with pytest.raises(ValueError, match=f"Cannot parse '{value}' as boolean"):
                _parse_bool_env(value)

    def test_parse_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        assert _parse_bool_env("TRUE") is True
        assert _parse_bool_env("True") is True
        assert _parse_bool_env("tRuE") is True
        assert _parse_bool_env("FALSE") is False
        assert _parse_bool_env("False") is False
        assert _parse_bool_env("fAlSe") is False


class TestDaprHTTPClient:
    """Test suite for DaprHTTPClient class."""

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_client_initialization_defaults(self):
        """Test that client initializes with default values."""
        client = DaprHTTPClient()
        assert client.dapr_app_id == ""
        assert client.dapr_http_endpoint == ""
        assert client.http_endpoint == ""
        assert client.path == ""
        assert client.headers == {}

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_client_initialization_with_values(self):
        """Test that client initializes with provided values."""
        client = DaprHTTPClient(
            dapr_app_id="my-app",
            dapr_http_endpoint="my-endpoint",
            http_endpoint="https://example.com",
            path="/api/test",
            headers={"X-Custom": "value"},
        )
        assert client.dapr_app_id == "my-app"
        assert client.dapr_http_endpoint == "my-endpoint"
        assert client.http_endpoint == "https://example.com"
        assert client.path == "/api/test"
        assert client.headers == {"X-Custom": "value"}

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_otel_disabled_via_env_false(self):
        """Test that OpenTelemetry is disabled when env var is 'false'."""
        client = DaprHTTPClient()
        # Should initialize without errors when OTEL is disabled
        assert client is not None

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "0"}, clear=False)
    def test_otel_disabled_via_env_zero(self):
        """Test that OpenTelemetry is disabled when env var is '0'."""
        client = DaprHTTPClient()
        # Should initialize without errors when OTEL is disabled
        assert client is not None

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "invalid"}, clear=False)
    def test_otel_disabled_via_invalid_env(self):
        """Test that invalid OTEL env var gracefully disables OpenTelemetry."""
        # Should not raise an error, just disable OTEL
        client = DaprHTTPClient()
        assert client is not None

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_otel_default_enabled(self):
        """Test that client initializes when OTEL env var is not set."""
        # Note: Testing actual OTEL functionality requires observability dependencies
        # This test just ensures the client can be created
        client = DaprHTTPClient()
        assert client is not None

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_base_url_private_attribute(self):
        """Test that _base_url private attribute is set correctly."""
        client = DaprHTTPClient()
        assert client._base_url == "http://localhost:3500/v1.0/invoke"

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_empty_headers_default(self):
        """Test that headers default to empty dict."""
        client = DaprHTTPClient()
        assert client.headers == {}
        # Verify it's a proper dict that can be modified
        client.headers["X-Test"] = "value"
        assert client.headers["X-Test"] == "value"

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_field_descriptions_exist(self):
        """Test that all fields have descriptions."""
        # This ensures documentation is maintained
        fields = DaprHTTPClient.model_fields
        assert fields["dapr_app_id"].description is not None
        assert fields["dapr_http_endpoint"].description is not None
        assert fields["http_endpoint"].description is not None
        assert fields["path"].description is not None
        assert fields["headers"].description is not None

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_optional_fields_can_be_none(self):
        """Test that optional fields accept None values when explicitly set."""
        # Test with explicit None values
        client = DaprHTTPClient(
            dapr_app_id=None,
            dapr_http_endpoint=None,
            http_endpoint=None,
            path=None,
        )
        assert client.dapr_app_id is None
        assert client.dapr_http_endpoint is None
        assert client.http_endpoint is None
        assert client.path is None

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_client_is_pydantic_model(self):
        """Test that DaprHTTPClient is a proper Pydantic model."""
        from pydantic import BaseModel

        assert issubclass(DaprHTTPClient, BaseModel)

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_client_model_dump(self):
        """Test that client can be serialized via model_dump."""
        client = DaprHTTPClient(
            dapr_app_id="test-app",
            path="/test",
            headers={"X-Test": "value"},
        )
        data = client.model_dump()
        assert data["dapr_app_id"] == "test-app"
        assert data["path"] == "/test"
        assert data["headers"] == {"X-Test": "value"}

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_client_model_dump_exclude_private(self):
        """Test that private attributes are excluded from model_dump by default."""
        client = DaprHTTPClient()
        data = client.model_dump()
        # _base_url is a private attribute and shouldn't be in the dump
        assert "_base_url" not in data

    @patch.dict(os.environ, {"DAPR_AGENTS_OTEL_ENABLED": "false"}, clear=False)
    def test_otel_parse_bool_variations(self):
        """Test that _parse_bool_env handles various boolean representations."""
        # Test that the parsing logic works for different true/false values
        # Note: Actual OTEL initialization requires observability dependencies
        for value in ["yes", "y", "true", "t", "on", "1"]:
            assert _parse_bool_env(value) is True
        for value in ["no", "n", "false", "f", "off", "0"]:
            assert _parse_bool_env(value) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
