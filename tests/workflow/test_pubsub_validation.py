"""Tests for PubSub component validation during subscription."""

import pytest
from unittest.mock import MagicMock

from dapr_agents.types.exceptions import PubSubNotAvailableError
from dapr_agents.workflow.utils.registration import _validate_pubsub_components


class TestPubSubNotAvailableError:
    """Tests for the PubSubNotAvailableError exception."""

    def test_exception_with_default_message(self):
        """Test exception creates a descriptive default message."""
        exc = PubSubNotAvailableError(pubsub_name="test-pubsub", topic="test-topic")

        assert exc.pubsub_name == "test-pubsub"
        assert exc.topic == "test-topic"
        assert "test-pubsub" in str(exc)
        assert "test-topic" in str(exc)
        assert "not available" in str(exc)

    def test_exception_with_custom_message(self):
        """Test exception can use a custom message."""
        custom_msg = "Custom error message"
        exc = PubSubNotAvailableError(
            pubsub_name="test-pubsub", topic="test-topic", message=custom_msg
        )

        assert exc.pubsub_name == "test-pubsub"
        assert exc.topic == "test-topic"
        assert str(exc) == custom_msg

    def test_exception_is_raised_and_caught(self):
        """Test that the exception can be raised and caught properly."""
        with pytest.raises(PubSubNotAvailableError) as exc_info:
            raise PubSubNotAvailableError(pubsub_name="my-pubsub", topic="my-topic")

        exc = exc_info.value
        assert exc.pubsub_name == "my-pubsub"
        assert exc.topic == "my-topic"


class TestValidatePubSubComponents:
    """Tests for _validate_pubsub_components function."""

    def _create_mock_dapr_client(self, pubsub_names: list[str]) -> MagicMock:
        """Helper to create a mock DaprClient with specific pubsub components."""
        mock_client = MagicMock()
        mock_metadata = MagicMock()

        components = []
        for name in pubsub_names:
            component = MagicMock()
            component.type = "pubsub.redis"
            component.name = name
            components.append(component)

        mock_metadata.registered_components = components
        mock_client.get_metadata.return_value = mock_metadata
        return mock_client

    def test_validation_passes_when_component_exists(self):
        """Test that validation passes when required pubsub component exists."""
        mock_client = self._create_mock_dapr_client(["agent-pubsub"])
        pubsub_names = {"agent-pubsub"}
        topics_by_pubsub = {"agent-pubsub": {"test-topic"}}

        # Should not raise
        _validate_pubsub_components(mock_client, pubsub_names, topics_by_pubsub)

    def test_validation_raises_when_component_missing(self):
        """Test that validation raises PubSubNotAvailableError when component is missing."""
        mock_client = self._create_mock_dapr_client(["other-pubsub"])
        pubsub_names = {"agent-pubsub"}
        topics_by_pubsub = {"agent-pubsub": {"test-topic"}}

        with pytest.raises(PubSubNotAvailableError) as exc_info:
            _validate_pubsub_components(mock_client, pubsub_names, topics_by_pubsub)

        exc = exc_info.value
        assert exc.pubsub_name == "agent-pubsub"
        assert exc.topic == "test-topic"

    def test_validation_with_multiple_pubsubs_all_exist(self):
        """Test validation passes when all required pubsubs exist."""
        mock_client = self._create_mock_dapr_client(["pubsub-a", "pubsub-b"])
        pubsub_names = {"pubsub-a", "pubsub-b"}
        topics_by_pubsub = {"pubsub-a": {"topic-a"}, "pubsub-b": {"topic-b"}}

        # Should not raise
        _validate_pubsub_components(mock_client, pubsub_names, topics_by_pubsub)

    def test_validation_with_multiple_pubsubs_one_missing(self):
        """Test validation raises when one of multiple pubsubs is missing."""
        mock_client = self._create_mock_dapr_client(["pubsub-a"])
        pubsub_names = {"pubsub-a", "pubsub-b"}
        topics_by_pubsub = {"pubsub-a": {"topic-a"}, "pubsub-b": {"topic-b"}}

        with pytest.raises(PubSubNotAvailableError) as exc_info:
            _validate_pubsub_components(mock_client, pubsub_names, topics_by_pubsub)

        exc = exc_info.value
        assert exc.pubsub_name == "pubsub-b"

    def test_validation_with_empty_pubsub_names(self):
        """Test validation does nothing when no pubsubs are required."""
        mock_client = MagicMock()
        pubsub_names: set[str] = set()
        topics_by_pubsub: dict[str, set[str]] = {}

        # Should not raise and should not call get_metadata
        _validate_pubsub_components(mock_client, pubsub_names, topics_by_pubsub)
        mock_client.get_metadata.assert_not_called()

    def test_validation_with_no_registered_components(self):
        """Test validation raises when no components are registered."""
        mock_client = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.registered_components = []
        mock_client.get_metadata.return_value = mock_metadata

        pubsub_names = {"agent-pubsub"}
        topics_by_pubsub = {"agent-pubsub": {"test-topic"}}

        with pytest.raises(PubSubNotAvailableError):
            _validate_pubsub_components(mock_client, pubsub_names, topics_by_pubsub)

    def test_validation_with_none_registered_components(self):
        """Test validation handles None registered_components gracefully."""
        mock_client = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.registered_components = None
        mock_client.get_metadata.return_value = mock_metadata

        pubsub_names = {"agent-pubsub"}
        topics_by_pubsub = {"agent-pubsub": {"test-topic"}}

        with pytest.raises(PubSubNotAvailableError):
            _validate_pubsub_components(mock_client, pubsub_names, topics_by_pubsub)

    def test_validation_logs_warning_on_metadata_error(self, caplog):
        """Test that metadata retrieval errors are logged but don't fail."""
        mock_client = MagicMock()
        mock_client.get_metadata.side_effect = Exception("Connection refused")

        pubsub_names = {"agent-pubsub"}
        topics_by_pubsub = {"agent-pubsub": {"test-topic"}}

        # Should not raise, but should log a warning
        _validate_pubsub_components(mock_client, pubsub_names, topics_by_pubsub)
        assert "Could not validate PubSub component availability" in caplog.text

    def test_validation_with_mixed_component_types(self):
        """Test validation only checks pubsub components, ignoring other types."""
        mock_client = MagicMock()
        mock_metadata = MagicMock()

        # Add various component types
        state_component = MagicMock()
        state_component.type = "state.redis"
        state_component.name = "agent-pubsub"  # Same name but wrong type

        pubsub_component = MagicMock()
        pubsub_component.type = "pubsub.redis"
        pubsub_component.name = "real-pubsub"

        mock_metadata.registered_components = [state_component, pubsub_component]
        mock_client.get_metadata.return_value = mock_metadata

        # Should raise because "agent-pubsub" is a state store, not a pubsub
        pubsub_names = {"agent-pubsub"}
        topics_by_pubsub = {"agent-pubsub": {"test-topic"}}

        with pytest.raises(PubSubNotAvailableError):
            _validate_pubsub_components(mock_client, pubsub_names, topics_by_pubsub)

    def test_validation_with_pubsub_type_variations(self):
        """Test validation handles different pubsub type strings."""
        mock_client = MagicMock()
        mock_metadata = MagicMock()

        # Test different pubsub type variations
        pubsub_component = MagicMock()
        pubsub_component.type = "pubsub.kafka"  # Different broker
        pubsub_component.name = "agent-pubsub"

        mock_metadata.registered_components = [pubsub_component]
        mock_client.get_metadata.return_value = mock_metadata

        pubsub_names = {"agent-pubsub"}
        topics_by_pubsub = {"agent-pubsub": {"test-topic"}}

        # Should not raise - "pubsub" is in the type string
        _validate_pubsub_components(mock_client, pubsub_names, topics_by_pubsub)
