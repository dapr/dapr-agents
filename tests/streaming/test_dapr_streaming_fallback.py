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

"""Tests for DaprChatClient streaming try-then-fallback behavior.

These tests exercise the `generate` method end-to-end with mocks substituted
for the Alpha2 gRPC client and the response normalizer, so we only cover the
stream-handling contract (no ValueError, fallback tagged in metadata).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dapr_agents.types.message import (
    AssistantMessage,
    LLMChatCandidate,
    LLMChatResponse,
)


def _fake_response() -> LLMChatResponse:
    return LLMChatResponse(
        results=[
            LLMChatCandidate(
                message=AssistantMessage(role="assistant", content="final answer"),
                finish_reason="stop",
            )
        ],
        metadata={"model": "dapr-test", "provider": "dapr"},
    )


def _make_client():
    """Construct a minimally-wired DaprChatClient that bypasses Dapr runtime calls."""

    # Importing lazily keeps the test isolated from Dapr side effects during collection.
    from dapr_agents.llm.dapr.chat import DaprChatClient

    fake_dapr = MagicMock()
    fake_dapr.get_metadata.return_value = MagicMock(
        registered_components=[], application_id="a"
    )
    fake_dapr.chat_completion_alpha2.return_value = {"dummy": "raw"}

    with patch(
        "dapr_agents.llm.dapr.client.DaprInferenceClientBase.get_client",
        return_value=fake_dapr,
    ):
        client = DaprChatClient(llm_component="conversation")
    return client, fake_dapr


@pytest.fixture
def client_and_dapr(monkeypatch):
    from dapr_agents.llm.dapr.chat import DaprChatClient

    client, fake_dapr = _make_client()
    # Keep the inference client mocked for the whole test, not just construction.
    # ``DaprInferenceClientBase.client`` is a non-caching property that calls
    # ``get_client()`` on every access, so ``generate()`` re-resolves it. Without a
    # persistent patch it builds a real ``DaprInferenceClient`` and hits the
    # suite-wide ``MockDaprClient`` (which has no ``converse_alpha2``).
    monkeypatch.setattr(
        "dapr_agents.llm.dapr.client.DaprInferenceClientBase.get_client",
        lambda self: fake_dapr,
    )
    monkeypatch.setattr(
        "dapr_agents.llm.dapr.chat.RequestHandler.normalize_chat_messages",
        lambda messages: messages,
    )
    monkeypatch.setattr(
        "dapr_agents.llm.dapr.chat.RequestHandler.process_params",
        lambda params, **_: params,
    )
    monkeypatch.setattr(
        "dapr_agents.llm.dapr.chat._check_dapr_runtime_support",
        lambda *_: None,
    )
    monkeypatch.setattr(
        DaprChatClient,
        "convert_to_conversation_inputs",
        lambda self, inputs: inputs,
    )
    monkeypatch.setattr(
        DaprChatClient,
        "translate_response",
        lambda self, raw, component: raw,
    )
    return client, fake_dapr


def test_stream_request_does_not_raise(client_and_dapr) -> None:
    client, _ = client_and_dapr
    with patch("dapr_agents.llm.dapr.chat.ResponseHandler.process_response") as mock_rp:
        mock_rp.return_value = _fake_response()
        result = client.generate(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )

    _, mock_kwargs = mock_rp.call_args
    # The stream kwarg was consumed inside `generate`; the response handler
    # is still driven synchronously.
    assert mock_kwargs["stream"] is False
    assert result.metadata.get("dapr_conversation_streaming_unsupported") is True


def test_non_stream_request_not_tagged(client_and_dapr) -> None:
    client, _ = client_and_dapr
    with patch("dapr_agents.llm.dapr.chat.ResponseHandler.process_response") as mock_rp:
        mock_rp.return_value = _fake_response()
        result = client.generate(messages=[{"role": "user", "content": "hi"}])
    assert "dapr_conversation_streaming_unsupported" not in (result.metadata or {})
