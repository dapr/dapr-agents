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


# ---------------------------------------------------------------------------
# options parameter merging tests
# ---------------------------------------------------------------------------


def test_options_param_no_user_options_uses_configured():
    """When no user options are passed, only configured options appear."""
    grpc_mock = MagicMock()
    shared = MagicMock()
    shared.get_grpc_channel = MagicMock()
    create_durabletask_module(shared)

    with patch.dict("sys.modules", {"grpc": grpc_mock}):
        opts = WorkflowGrpcOptions(max_send_message_length=4 * 1024 * 1024)
        apply_grpc_options(opts)

        shared.get_grpc_channel("localhost:4001", options=None)
        call_kwargs = grpc_mock.insecure_channel.call_args.kwargs
        options = dict(call_kwargs["options"])
        assert options["grpc.max_send_message_length"] == 4 * 1024 * 1024


def test_options_param_user_options_merged():
    """User-provided options are merged with configured options."""
    grpc_mock = MagicMock()
    shared = MagicMock()
    shared.get_grpc_channel = MagicMock()
    create_durabletask_module(shared)

    with patch.dict("sys.modules", {"grpc": grpc_mock}):
        opts = WorkflowGrpcOptions(max_send_message_length=4 * 1024 * 1024)
        apply_grpc_options(opts)

        user_options = [("grpc.initial_reconnect_backoff_ms", 500)]
        shared.get_grpc_channel("localhost:4001", options=user_options)
        call_kwargs = grpc_mock.insecure_channel.call_args.kwargs
        options = dict(call_kwargs["options"])
        assert options["grpc.max_send_message_length"] == 4 * 1024 * 1024
        assert options["grpc.initial_reconnect_backoff_ms"] == 500


def test_options_param_user_options_override_configured():
    """User-provided options override configured options for duplicate keys."""
    grpc_mock = MagicMock()
    shared = MagicMock()
    shared.get_grpc_channel = MagicMock()
    create_durabletask_module(shared)

    with patch.dict("sys.modules", {"grpc": grpc_mock}):
        opts = WorkflowGrpcOptions(max_send_message_length=4 * 1024 * 1024)
        apply_grpc_options(opts)

        user_options = [("grpc.max_send_message_length", 99 * 1024 * 1024)]
        shared.get_grpc_channel("localhost:4001", options=user_options)
        call_kwargs = grpc_mock.insecure_channel.call_args.kwargs
        options = dict(call_kwargs["options"])
        assert options["grpc.max_send_message_length"] == 99 * 1024 * 1024


# ---------------------------------------------------------------------------
# Keepalive field tests
# ---------------------------------------------------------------------------


def test_keepalive_fields_defaults():
    """Keepalive fields default to None."""
    opts = WorkflowGrpcOptions()
    assert opts.keepalive_time_ms is None
    assert opts.keepalive_timeout_ms is None


def test_keepalive_time_ms_validation_zero():
    with pytest.raises(ValueError, match="keepalive_time_ms must be greater than 0"):
        WorkflowGrpcOptions(keepalive_time_ms=0)


def test_keepalive_time_ms_validation_negative():
    with pytest.raises(ValueError, match="keepalive_time_ms must be greater than 0"):
        WorkflowGrpcOptions(keepalive_time_ms=-1)


def test_keepalive_timeout_ms_validation_zero():
    with pytest.raises(ValueError, match="keepalive_timeout_ms must be greater than 0"):
        WorkflowGrpcOptions(keepalive_timeout_ms=0)


def test_keepalive_timeout_ms_validation_negative():
    with pytest.raises(ValueError, match="keepalive_timeout_ms must be greater than 0"):
        WorkflowGrpcOptions(keepalive_timeout_ms=-1)


def test_keepalive_valid_values():
    opts = WorkflowGrpcOptions(keepalive_time_ms=30000, keepalive_timeout_ms=5000)
    assert opts.keepalive_time_ms == 30000
    assert opts.keepalive_timeout_ms == 5000


def test_apply_grpc_options_keepalive_only():
    """Keepalive-only options should still trigger patching."""
    grpc_mock = MagicMock()
    shared = MagicMock()
    original = MagicMock()
    shared.get_grpc_channel = original
    create_durabletask_module(shared)

    with patch.dict("sys.modules", {"grpc": grpc_mock}):
        opts = WorkflowGrpcOptions(keepalive_time_ms=30000, keepalive_timeout_ms=5000)
        apply_grpc_options(opts)

        assert shared.get_grpc_channel is not original
        shared.get_grpc_channel("localhost:4001")
        call_kwargs = grpc_mock.insecure_channel.call_args.kwargs
        options = dict(call_kwargs["options"])
        assert options["grpc.keepalive_time_ms"] == 30000
        assert options["grpc.keepalive_timeout_ms"] == 5000
        assert "grpc.max_send_message_length" not in options
        assert "grpc.max_receive_message_length" not in options


def test_apply_grpc_options_all_fields():
    """All four fields together produce the correct channel options."""
    grpc_mock = MagicMock()
    shared = MagicMock()
    shared.get_grpc_channel = MagicMock()
    create_durabletask_module(shared)

    with patch.dict("sys.modules", {"grpc": grpc_mock}):
        opts = WorkflowGrpcOptions(
            max_send_message_length=8 * 1024 * 1024,
            max_receive_message_length=12 * 1024 * 1024,
            keepalive_time_ms=60000,
            keepalive_timeout_ms=10000,
        )
        apply_grpc_options(opts)

        shared.get_grpc_channel("localhost:4001")
        call_kwargs = grpc_mock.insecure_channel.call_args.kwargs
        options = dict(call_kwargs["options"])
        assert options["grpc.max_send_message_length"] == 8 * 1024 * 1024
        assert options["grpc.max_receive_message_length"] == 12 * 1024 * 1024
        assert options["grpc.keepalive_time_ms"] == 60000
        assert options["grpc.keepalive_timeout_ms"] == 10000


def test_apply_grpc_options_no_patch_when_all_none():
    """No patching when all fields are None (early return)."""
    shared = MagicMock()
    original = MagicMock()
    shared.get_grpc_channel = original
    create_durabletask_module(shared)

    with patch.dict("sys.modules", {"grpc": MagicMock()}):
        opts = WorkflowGrpcOptions()
        apply_grpc_options(opts)
        assert shared.get_grpc_channel is original
