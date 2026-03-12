"""Tests for PubSub component validation during subscription."""

import pytest
from unittest.mock import MagicMock, patch

from dapr_agents.types.exceptions import PubSubNotAvailableError
from dapr_agents.workflow.utils.registration import _validate_pubsub_components

_PATCH_TARGET = "dapr_agents.workflow.utils.registration.DaprClient"


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


def _mock_dapr_client_with_pubsubs(pubsub_names: list[str]) -> MagicMock:
    """Create a mock DaprClient constructor that returns a context-manager mock
    whose ``get_metadata()`` reports the given pubsub component names."""
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
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    # The constructor mock — calling DaprClient() returns mock_client
    mock_cls = MagicMock(return_value=mock_client)
    return mock_cls, mock_client


class TestValidatePubSubComponents:
    """Tests for _validate_pubsub_components function."""

    def test_validation_passes_when_component_exists(self):
        """Test that validation passes when required pubsub component exists."""
        mock_cls, _ = _mock_dapr_client_with_pubsubs(["agent-pubsub"])
        with patch(_PATCH_TARGET, mock_cls):
            _validate_pubsub_components(
                MagicMock(), {"agent-pubsub"}, {"agent-pubsub": {"test-topic"}}
            )

    def test_validation_raises_when_component_missing(self):
        """Test that validation raises PubSubNotAvailableError when component is missing."""
        mock_cls, _ = _mock_dapr_client_with_pubsubs(["other-pubsub"])
        with patch(_PATCH_TARGET, mock_cls):
            with pytest.raises(PubSubNotAvailableError) as exc_info:
                _validate_pubsub_components(
                    MagicMock(), {"agent-pubsub"}, {"agent-pubsub": {"test-topic"}}
                )
            exc = exc_info.value
            assert exc.pubsub_name == "agent-pubsub"
            assert exc.topic == "test-topic"

    def test_validation_with_multiple_pubsubs_all_exist(self):
        """Test validation passes when all required pubsubs exist."""
        mock_cls, _ = _mock_dapr_client_with_pubsubs(["pubsub-a", "pubsub-b"])
        with patch(_PATCH_TARGET, mock_cls):
            _validate_pubsub_components(
                MagicMock(),
                {"pubsub-a", "pubsub-b"},
                {"pubsub-a": {"topic-a"}, "pubsub-b": {"topic-b"}},
            )

    def test_validation_with_multiple_pubsubs_one_missing(self):
        """Test validation raises when one of multiple pubsubs is missing."""
        mock_cls, _ = _mock_dapr_client_with_pubsubs(["pubsub-a"])
        with patch(_PATCH_TARGET, mock_cls):
            with pytest.raises(PubSubNotAvailableError) as exc_info:
                _validate_pubsub_components(
                    MagicMock(),
                    {"pubsub-a", "pubsub-b"},
                    {"pubsub-a": {"topic-a"}, "pubsub-b": {"topic-b"}},
                )
            assert exc_info.value.pubsub_name == "pubsub-b"

    def test_validation_with_empty_pubsub_names(self):
        """Test validation does nothing when no pubsubs are required."""
        # Empty pubsub_names → early return, no DaprClient created
        _validate_pubsub_components(MagicMock(), set(), {})

    def test_validation_with_no_registered_components(self):
        """Test validation raises when no components are registered."""
        mock_cls, _ = _mock_dapr_client_with_pubsubs([])
        with patch(_PATCH_TARGET, mock_cls):
            with pytest.raises(PubSubNotAvailableError):
                _validate_pubsub_components(
                    MagicMock(), {"agent-pubsub"}, {"agent-pubsub": {"test-topic"}}
                )

    def test_validation_with_none_registered_components(self):
        """Test validation handles None registered_components gracefully."""
        mock_cls, mock_client = _mock_dapr_client_with_pubsubs([])
        mock_client.get_metadata.return_value.registered_components = None
        with patch(_PATCH_TARGET, mock_cls):
            with pytest.raises(PubSubNotAvailableError):
                _validate_pubsub_components(
                    MagicMock(), {"agent-pubsub"}, {"agent-pubsub": {"test-topic"}}
                )

    def test_validation_raises_on_metadata_error(self, caplog):
        """Test that metadata retrieval errors are logged and raise an exception."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get_metadata.side_effect = Exception("Connection refused")
        mock_cls = MagicMock(return_value=mock_client)

        with patch(_PATCH_TARGET, mock_cls):
            with pytest.raises(Exception, match="Connection refused"):
                _validate_pubsub_components(
                    MagicMock(), {"agent-pubsub"}, {"agent-pubsub": {"test-topic"}}
                )
        assert "Could not validate PubSub component availability" in caplog.text
        assert "Failing startup to prevent silent subscription failures" in caplog.text

    def test_validation_with_mixed_component_types(self):
        """Test validation only checks pubsub components, ignoring other types."""
        mock_client = MagicMock()
        mock_metadata = MagicMock()

        state_component = MagicMock()
        state_component.type = "state.redis"
        state_component.name = "agent-pubsub"  # Same name but wrong type

        pubsub_component = MagicMock()
        pubsub_component.type = "pubsub.redis"
        pubsub_component.name = "real-pubsub"

        mock_metadata.registered_components = [state_component, pubsub_component]
        mock_client.get_metadata.return_value = mock_metadata
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_cls = MagicMock(return_value=mock_client)

        with patch(_PATCH_TARGET, mock_cls):
            with pytest.raises(PubSubNotAvailableError):
                _validate_pubsub_components(
                    MagicMock(), {"agent-pubsub"}, {"agent-pubsub": {"test-topic"}}
                )

    def test_validation_with_pubsub_type_variations(self):
        """Test validation handles different pubsub type strings."""
        mock_client = MagicMock()
        mock_metadata = MagicMock()

        pubsub_component = MagicMock()
        pubsub_component.type = "pubsub.kafka"
        pubsub_component.name = "agent-pubsub"

        mock_metadata.registered_components = [pubsub_component]
        mock_client.get_metadata.return_value = mock_metadata
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_cls = MagicMock(return_value=mock_client)

        with patch(_PATCH_TARGET, mock_cls):
            _validate_pubsub_components(
                MagicMock(), {"agent-pubsub"}, {"agent-pubsub": {"test-topic"}}
            )
