"""Tests for gRPC configuration in WorkflowApp."""
import pytest
from unittest.mock import MagicMock, patch, call
import types
from dapr_agents.workflow.base import WorkflowApp


@pytest.fixture
def mock_workflow_dependencies():
    """Mock all the dependencies needed for WorkflowApp initialization."""
    with patch("dapr_agents.workflow.base.WorkflowRuntime") as mock_runtime, patch(
        "dapr_agents.workflow.base.DaprWorkflowClient"
    ) as mock_client, patch(
        "dapr_agents.workflow.base.get_default_llm"
    ) as mock_llm, patch.object(
        WorkflowApp, "start_runtime"
    ) as mock_start, patch.object(
        WorkflowApp, "setup_signal_handlers"
    ) as mock_handlers:
        mock_runtime_instance = MagicMock()
        mock_runtime.return_value = mock_runtime_instance

        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        yield {
            "runtime": mock_runtime,
            "runtime_instance": mock_runtime_instance,
            "client": mock_client,
            "client_instance": mock_client_instance,
            "llm": mock_llm,
            "llm_instance": mock_llm_instance,
            "start_runtime": mock_start,
            "signal_handlers": mock_handlers,
        }


def test_workflow_app_without_grpc_config(mock_workflow_dependencies):
    """Test that WorkflowApp initializes without gRPC configuration."""
    # Create WorkflowApp without gRPC config
    app = WorkflowApp()

    # Verify the app was created
    assert app is not None
    assert app.grpc_max_send_message_length is None
    assert app.grpc_max_receive_message_length is None

    # Verify runtime and client were initialized
    assert app.wf_runtime is not None
    assert app.wf_client is not None


def test_workflow_app_with_grpc_config(mock_workflow_dependencies):
    """Test that WorkflowApp initializes with gRPC configuration."""
    # Mock the grpc module and durabletask shared module
    mock_grpc = MagicMock()
    mock_shared = MagicMock()
    mock_channel = MagicMock()

    # Set up the mock channel
    mock_grpc.insecure_channel.return_value = mock_channel
    mock_shared.get_grpc_channel = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "grpc": mock_grpc,
            "durabletask.internal.shared": mock_shared,
        },
    ):
        # Create WorkflowApp with gRPC config (16MB)
        app = WorkflowApp(
            grpc_max_send_message_length=16 * 1024 * 1024,  # 16MB
            grpc_max_receive_message_length=16 * 1024 * 1024,  # 16MB
        )

        # Verify the configuration was set
        assert app.grpc_max_send_message_length == 16 * 1024 * 1024
        assert app.grpc_max_receive_message_length == 16 * 1024 * 1024

        # Verify runtime and client were initialized
        assert app.wf_runtime is not None
        assert app.wf_client is not None


def test_configure_grpc_channel_options_is_called(mock_workflow_dependencies):
    """Test that _configure_grpc_channel_options is called when gRPC config is provided."""
    with patch.object(WorkflowApp, "_configure_grpc_channel_options") as mock_configure:
        # Create WorkflowApp with gRPC config
        WorkflowApp(
            grpc_max_send_message_length=8 * 1024 * 1024,  # 8MB
        )

        # Verify the configuration method was called
        mock_configure.assert_called_once()


def test_configure_grpc_channel_options_not_called_without_config(
    mock_workflow_dependencies,
):
    """Test that _configure_grpc_channel_options is not called without gRPC config."""
    with patch.object(WorkflowApp, "_configure_grpc_channel_options") as mock_configure:
        # Create WorkflowApp without gRPC config
        WorkflowApp()

        # Verify the configuration method was NOT called
        mock_configure.assert_not_called()


def test_grpc_channel_patching():
    """Test that the gRPC channel factory is properly patched with custom options."""
    # Mock the grpc module and durabletask shared module
    mock_grpc = MagicMock()
    mock_shared = MagicMock()
    mock_channel = MagicMock()

    # Set up the mock channel
    mock_grpc.insecure_channel.return_value = mock_channel

    # Keep original reference
    def original_get_grpc_channel(*_, **__):
        return "original"

    mock_shared.get_grpc_channel = original_get_grpc_channel

    # Create dummy package/module structure so 'from durabletask.internal import shared' works
    durabletask_module = types.ModuleType("durabletask")
    internal_module = types.ModuleType("durabletask.internal")
    setattr(durabletask_module, "internal", internal_module)
    setattr(internal_module, "shared", mock_shared)

    with patch.dict(
        "sys.modules",
        {
            "grpc": mock_grpc,
            "durabletask": durabletask_module,
            "durabletask.internal": internal_module,
            "durabletask.internal.shared": mock_shared,
        },
    ), patch("dapr_agents.workflow.base.WorkflowRuntime"), patch(
        "dapr_agents.workflow.base.DaprWorkflowClient"
    ), patch("dapr_agents.workflow.base.get_default_llm"), patch.object(
        WorkflowApp, "start_runtime"
    ), patch.object(WorkflowApp, "setup_signal_handlers"):
        # Create WorkflowApp with gRPC config
        max_send = 10 * 1024 * 1024  # 10MB
        max_recv = 12 * 1024 * 1024  # 12MB

        WorkflowApp(
            grpc_max_send_message_length=max_send,
            grpc_max_receive_message_length=max_recv,
        )

        # Confirm get_grpc_channel was overridden
        assert callable(mock_shared.get_grpc_channel)
        assert mock_shared.get_grpc_channel is not original_get_grpc_channel
        assert (
            getattr(mock_shared.get_grpc_channel, "__name__", "")
            == "get_grpc_channel_with_options"
        )

        # Call the patched function
        test_address = "localhost:50001"
        mock_shared.get_grpc_channel(test_address)

        # Verify insecure_channel was called with correct options
        mock_grpc.insecure_channel.assert_called_once()
        call_args = mock_grpc.insecure_channel.call_args

        # Check that the address was passed
        assert call_args[0][0] == test_address

        # Check that options were passed
        assert "options" in call_args.kwargs
        options = call_args.kwargs["options"]

        # Verify options contain our custom message size limits
        assert ("grpc.max_send_message_length", max_send) in options
        assert ("grpc.max_receive_message_length", max_recv) in options


def test_grpc_config_with_only_send_limit(mock_workflow_dependencies):
    """Test gRPC configuration with only send limit set."""
    with patch.object(WorkflowApp, "_configure_grpc_channel_options") as mock_configure:
        app = WorkflowApp(
            grpc_max_send_message_length=20 * 1024 * 1024,  # 20MB
        )

        # Verify configuration was called
        mock_configure.assert_called_once()

        # Verify only send limit was set
        assert app.grpc_max_send_message_length == 20 * 1024 * 1024
        assert app.grpc_max_receive_message_length is None


def test_grpc_config_with_only_receive_limit(mock_workflow_dependencies):
    """Test gRPC configuration with only receive limit set."""
    with patch.object(WorkflowApp, "_configure_grpc_channel_options") as mock_configure:
        app = WorkflowApp(
            grpc_max_receive_message_length=24 * 1024 * 1024,  # 24MB
        )

        # Verify configuration was called
        mock_configure.assert_called_once()

        # Verify only receive limit was set
        assert app.grpc_max_send_message_length is None
        assert app.grpc_max_receive_message_length == 24 * 1024 * 1024
