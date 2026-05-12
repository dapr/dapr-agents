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

import logging
import os
from unittest import mock

import pytest

from dapr_agents.utils.dapr_client_factory import (
    INBOUND_MESSAGE_SIZE_ENV,
    dapr_client_kwargs,
    get_inbound_message_size_bytes,
    set_inbound_message_size_bytes,
)


@pytest.fixture(autouse=True)
def _clear_state() -> None:
    """Reset env var and programmatic override around every test."""
    set_inbound_message_size_bytes(None)
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(INBOUND_MESSAGE_SIZE_ENV, None)
        yield
    set_inbound_message_size_bytes(None)


def test_no_env_returns_explicit_kwargs_unchanged() -> None:
    result = dapr_client_kwargs(http_timeout_seconds=10)
    assert result == {"http_timeout_seconds": 10}


def test_no_env_with_no_explicit_returns_empty() -> None:
    assert dapr_client_kwargs() == {}


def test_env_set_adds_max_grpc_message_length() -> None:
    with mock.patch.dict(os.environ, {INBOUND_MESSAGE_SIZE_ENV: str(16 * 1024 * 1024)}):
        result = dapr_client_kwargs()

    assert result == {"max_grpc_message_length": 16 * 1024 * 1024}


def test_env_set_preserves_other_explicit_kwargs() -> None:
    with mock.patch.dict(os.environ, {INBOUND_MESSAGE_SIZE_ENV: "8388608"}):
        result = dapr_client_kwargs(http_timeout_seconds=10)

    assert result == {
        "http_timeout_seconds": 10,
        "max_grpc_message_length": 8388608,
    }


def test_explicit_max_grpc_message_length_wins_over_env() -> None:
    with mock.patch.dict(os.environ, {INBOUND_MESSAGE_SIZE_ENV: "8388608"}):
        result = dapr_client_kwargs(max_grpc_message_length=64 * 1024 * 1024)

    assert result == {"max_grpc_message_length": 64 * 1024 * 1024}


def test_invalid_env_logs_warning_and_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        mock.patch.dict(os.environ, {INBOUND_MESSAGE_SIZE_ENV: "not-a-number"}),
        caplog.at_level(
            logging.WARNING, logger="dapr_agents.utils.dapr_client_factory"
        ),
    ):
        result = dapr_client_kwargs()

    assert result == {}
    assert "Ignoring invalid" in caplog.text
    assert INBOUND_MESSAGE_SIZE_ENV in caplog.text


def test_returned_kwargs_is_independent_of_explicit() -> None:
    """The helper must not mutate the caller's kwargs dict."""
    explicit = {"http_timeout_seconds": 10}
    result = dapr_client_kwargs(**explicit)
    result["http_timeout_seconds"] = 999
    assert explicit == {"http_timeout_seconds": 10}


def test_setter_sets_override_consumed_by_kwargs() -> None:
    set_inbound_message_size_bytes(16 * 1024 * 1024)
    assert get_inbound_message_size_bytes() == 16 * 1024 * 1024
    assert dapr_client_kwargs() == {"max_grpc_message_length": 16 * 1024 * 1024}


def test_setter_override_beats_env_var() -> None:
    with mock.patch.dict(os.environ, {INBOUND_MESSAGE_SIZE_ENV: str(4 * 1024 * 1024)}):
        set_inbound_message_size_bytes(32 * 1024 * 1024)
        result = dapr_client_kwargs()

    assert result == {"max_grpc_message_length": 32 * 1024 * 1024}


def test_explicit_kwarg_beats_setter_override() -> None:
    set_inbound_message_size_bytes(16 * 1024 * 1024)
    result = dapr_client_kwargs(max_grpc_message_length=64 * 1024 * 1024)

    assert result == {"max_grpc_message_length": 64 * 1024 * 1024}


def test_setter_none_clears_override_falls_through_to_env() -> None:
    set_inbound_message_size_bytes(16 * 1024 * 1024)
    set_inbound_message_size_bytes(None)
    with mock.patch.dict(os.environ, {INBOUND_MESSAGE_SIZE_ENV: "8388608"}):
        result = dapr_client_kwargs()

    assert result == {"max_grpc_message_length": 8388608}


def test_setter_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError):
        set_inbound_message_size_bytes(0)
    with pytest.raises(ValueError):
        set_inbound_message_size_bytes(-1)
