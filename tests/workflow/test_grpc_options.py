from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from dapr_agents.agents.configs import WorkflowGrpcOptions
from dapr_agents.workflow.utils.grpc import apply_grpc_options


def create_durabletask_module(shared_module: MagicMock) -> None:
    durabletask_module = types.ModuleType("durabletask")
    internal_module = types.ModuleType("durabletask.internal")
    setattr(durabletask_module, "internal", internal_module)
    setattr(internal_module, "shared", shared_module)
    import sys

    sys.modules["durabletask"] = durabletask_module
    sys.modules["durabletask.internal"] = internal_module
    sys.modules["durabletask.internal.shared"] = shared_module


@pytest.fixture(autouse=True)
def cleanup_modules():
    import sys

    snapshot = sys.modules.copy()
    yield
    for key in list(sys.modules.keys()):
        if key not in snapshot:
            del sys.modules[key]


def test_apply_grpc_options_no_options():
    shared = MagicMock()
    original = MagicMock()
    shared.get_grpc_channel = original
    create_durabletask_module(shared)
    with patch.dict("sys.modules", {"grpc": MagicMock()}):
        apply_grpc_options(None)
        assert shared.get_grpc_channel is original


def test_apply_grpc_options_only_send():
    grpc_mock = MagicMock()
    shared = MagicMock()
    shared.get_grpc_channel = MagicMock()
    create_durabletask_module(shared)
    with patch.dict("sys.modules", {"grpc": grpc_mock}):
        opts = WorkflowGrpcOptions(max_send_message_length=16 * 1024 * 1024)
        apply_grpc_options(opts)

        assert callable(shared.get_grpc_channel)
        shared.get_grpc_channel("localhost:4001")
        grpc_mock.insecure_channel.assert_called_once()
        call_kwargs = grpc_mock.insecure_channel.call_args.kwargs
        assert ("grpc.max_send_message_length", 16 * 1024 * 1024) in call_kwargs[
            "options"
        ]
        assert "grpc.max_receive_message_length" not in dict(call_kwargs["options"])


def test_apply_grpc_options_only_receive():
    grpc_mock = MagicMock()
    shared = MagicMock()
    shared.get_grpc_channel = MagicMock()
    create_durabletask_module(shared)
    with patch.dict("sys.modules", {"grpc": grpc_mock}):
        opts = WorkflowGrpcOptions(max_receive_message_length=24 * 1024 * 1024)
        apply_grpc_options(opts)

        shared.get_grpc_channel("localhost:4001")
        grpc_mock.insecure_channel.assert_called_once()
        call_kwargs = grpc_mock.insecure_channel.call_args.kwargs
        assert ("grpc.max_receive_message_length", 24 * 1024 * 1024) in call_kwargs[
            "options"
        ]
        assert "grpc.max_send_message_length" not in dict(call_kwargs["options"])


def test_apply_grpc_options_patch_occurs():
    grpc_mock = MagicMock()
    shared = MagicMock()
    original = MagicMock()
    shared.get_grpc_channel = original
    create_durabletask_module(shared)

    with patch.dict("sys.modules", {"grpc": grpc_mock}):
        opts = WorkflowGrpcOptions(
            max_send_message_length=8 * 1024 * 1024,
            max_receive_message_length=12 * 1024 * 1024,
        )
        apply_grpc_options(opts)

        assert callable(shared.get_grpc_channel)
        assert shared.get_grpc_channel is not original
        shared.get_grpc_channel("localhost:50001")

        grpc_mock.insecure_channel.assert_called_once()
        kwargs = grpc_mock.insecure_channel.call_args.kwargs
        options = dict(kwargs["options"])
        assert options["grpc.max_send_message_length"] == 8 * 1024 * 1024
        assert options["grpc.max_receive_message_length"] == 12 * 1024 * 1024
