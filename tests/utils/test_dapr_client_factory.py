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
    DaprClientConfig,
    dapr_client_kwargs,
)


@pytest.fixture(autouse=True)
def _clear_env() -> None:
    """Ensure the inbound size env var is unset around every test."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(INBOUND_MESSAGE_SIZE_ENV, None)
        yield


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


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_non_positive_env_logs_warning_and_returns_empty(
    raw: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        mock.patch.dict(os.environ, {INBOUND_MESSAGE_SIZE_ENV: raw}),
        caplog.at_level(
            logging.WARNING, logger="dapr_agents.utils.dapr_client_factory"
        ),
    ):
        result = dapr_client_kwargs()

    assert result == {}
    assert "Ignoring non-positive" in caplog.text
    assert INBOUND_MESSAGE_SIZE_ENV in caplog.text


def test_returned_kwargs_is_independent_of_explicit() -> None:
    """The helper must not mutate the caller's kwargs dict."""
    explicit = {"http_timeout_seconds": 10}
    result = dapr_client_kwargs(**explicit)
    result["http_timeout_seconds"] = 999
    assert explicit == {"http_timeout_seconds": 10}


def test_config_supplies_max_grpc_message_length() -> None:
    config = DaprClientConfig(max_grpc_message_length=16 * 1024 * 1024)
    assert dapr_client_kwargs(config=config) == {
        "max_grpc_message_length": 16 * 1024 * 1024
    }


def test_config_with_none_falls_through_to_env() -> None:
    config = DaprClientConfig()
    with mock.patch.dict(os.environ, {INBOUND_MESSAGE_SIZE_ENV: "8388608"}):
        result = dapr_client_kwargs(config=config)

    assert result == {"max_grpc_message_length": 8388608}


def test_config_beats_env() -> None:
    config = DaprClientConfig(max_grpc_message_length=32 * 1024 * 1024)
    with mock.patch.dict(os.environ, {INBOUND_MESSAGE_SIZE_ENV: str(4 * 1024 * 1024)}):
        result = dapr_client_kwargs(config=config)

    assert result == {"max_grpc_message_length": 32 * 1024 * 1024}


def test_explicit_kwarg_beats_config() -> None:
    config = DaprClientConfig(max_grpc_message_length=16 * 1024 * 1024)
    result = dapr_client_kwargs(config=config, max_grpc_message_length=64 * 1024 * 1024)
    assert result == {"max_grpc_message_length": 64 * 1024 * 1024}


@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValueError):
        DaprClientConfig(max_grpc_message_length=value)


def test_dapr_client_config_is_immutable() -> None:
    config = DaprClientConfig(max_grpc_message_length=16 * 1024 * 1024)
    with pytest.raises(Exception):
        config.max_grpc_message_length = 32 * 1024 * 1024  # type: ignore[misc]


def test_explicit_kwarg_none_is_dropped_and_falls_through_to_env() -> None:
    """An explicit ``max_grpc_message_length=None`` must not short-circuit resolution."""
    with mock.patch.dict(os.environ, {INBOUND_MESSAGE_SIZE_ENV: "8388608"}):
        result = dapr_client_kwargs(max_grpc_message_length=None)

    assert result == {"max_grpc_message_length": 8388608}


def test_explicit_kwarg_none_with_config_uses_config() -> None:
    """When the explicit kwarg is None, the config (if present) wins over env."""
    config = DaprClientConfig(max_grpc_message_length=32 * 1024 * 1024)
    with mock.patch.dict(os.environ, {INBOUND_MESSAGE_SIZE_ENV: "8388608"}):
        result = dapr_client_kwargs(config=config, max_grpc_message_length=None)

    assert result == {"max_grpc_message_length": 32 * 1024 * 1024}


def test_explicit_kwarg_none_with_no_other_source_returns_empty() -> None:
    assert dapr_client_kwargs(max_grpc_message_length=None) == {}


@pytest.mark.parametrize("value", [0, -1])
def test_explicit_kwarg_rejects_non_positive_int(value: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        dapr_client_kwargs(max_grpc_message_length=value)
