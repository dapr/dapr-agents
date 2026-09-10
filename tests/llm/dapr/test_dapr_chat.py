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

"""Tests for DaprChatClient token-usage propagation.

The Alpha2 Conversation API reports per-output token usage. These tests pin
the contract that real usage survives both conversion layers — the gRPC
dataclass→dict flattening in ``DaprInferenceClient.chat_completion_alpha2``
and the OpenAI-style envelope built by ``DaprChatClient.translate_response``
— as ints, and that no hardcoded sentinel (the old ``{"total_tokens": "-1"}``)
can come back when the runtime reports nothing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest


def _raw_response(usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the dict shape ``chat_completion_alpha2`` hands to ``translate_response``."""
    output: Dict[str, Any] = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ]
    }
    if usage is not None:
        output["usage"] = usage
    return {"context_id": None, "outputs": [output]}


def _make_client():
    """Construct a minimally-wired DaprChatClient that bypasses Dapr runtime calls."""
    # Importing lazily keeps the test isolated from Dapr side effects during collection.
    from dapr_agents.llm.dapr.chat import DaprChatClient

    fake_dapr = MagicMock()
    fake_dapr.get_metadata.return_value = MagicMock(
        registered_components=[], application_id="a"
    )

    with patch(
        "dapr_agents.llm.dapr.client.DaprInferenceClientBase.get_client",
        return_value=fake_dapr,
    ):
        client = DaprChatClient(component_name="conversation")
    return client, fake_dapr


@pytest.fixture
def dapr_chat_client():
    client, _ = _make_client()
    return client


class TestTranslateResponseUsage:
    def test_translate_response_propagates_usage_as_ints(
        self, dapr_chat_client
    ) -> None:
        usage = {
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "total_tokens": 46,
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 8},
        }
        envelope = dapr_chat_client.translate_response(
            _raw_response(usage=usage), "echo"
        )
        assert envelope["usage"] == usage
        # A copy rides in the envelope — the raw response dict is not aliased.
        assert envelope["usage"] is not usage
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert isinstance(envelope["usage"][field], int)

    def test_translate_response_coerces_stray_string_counters(
        self, dapr_chat_client
    ) -> None:
        # Defense in depth: even if a conversion layer regresses to string
        # values, the envelope still carries ints.
        usage = {"prompt_tokens": "12", "completion_tokens": "34", "total_tokens": "46"}
        envelope = dapr_chat_client.translate_response(
            _raw_response(usage=usage), "echo"
        )
        assert envelope["usage"] == {
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "total_tokens": 46,
        }

    def test_translate_response_omits_usage_when_unreported(
        self, dapr_chat_client
    ) -> None:
        # Anti-sentinel regression: the old code emitted {"total_tokens": "-1"}
        # unconditionally; without runtime-reported usage the key must be absent.
        envelope = dapr_chat_client.translate_response(_raw_response(), "echo")
        assert "usage" not in envelope

    def test_translate_response_sums_usage_across_outputs(
        self, dapr_chat_client
    ) -> None:
        raw = {
            "context_id": None,
            "outputs": [
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "a"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                },
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "b"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 5,
                        "total_tokens": 7,
                    },
                },
            ],
        }
        envelope = dapr_chat_client.translate_response(raw, "echo")
        assert envelope["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 7,
            "total_tokens": 10,
        }


class TestChatCompletionAlpha2Usage:
    """The gRPC-layer conversion must not drop per-output usage or model."""

    def _client_with_response(self, response: Any):
        from dapr_agents.llm.dapr.client import DaprInferenceClient

        grpc_client = MagicMock()
        grpc_client.__enter__.return_value = grpc_client
        grpc_client.__exit__.return_value = False
        grpc_client.converse_alpha2.return_value = response
        return DaprInferenceClient(client_factory=lambda: grpc_client)

    def test_chat_completion_alpha2_emits_usage_and_model(self) -> None:
        output = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="gpt-4o-mini-2024-07-18",
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                completion_tokens_details=None,
                prompt_tokens_details=SimpleNamespace(audio_tokens=0, cached_tokens=3),
            ),
        )
        client = self._client_with_response(
            SimpleNamespace(context_id="ctx-1", outputs=[output])
        )

        result = client.chat_completion_alpha2(llm="echo", inputs=[])

        assert result["context_id"] == "ctx-1"
        assert result["outputs"][0]["model"] == "gpt-4o-mini-2024-07-18"
        assert result["outputs"][0]["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 3},
        }
        # Absent detail breakdowns stay absent rather than riding along as None.
        assert "completion_tokens_details" not in result["outputs"][0]["usage"]

    def test_chat_completion_alpha2_degrades_without_usage(self) -> None:
        # Older runtimes report no usage/model; the envelope must omit the keys
        # instead of inventing sentinels.
        output = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model=None,
            usage=None,
        )
        client = self._client_with_response(
            SimpleNamespace(context_id=None, outputs=[output])
        )

        result = client.chat_completion_alpha2(llm="echo", inputs=[])

        assert "usage" not in result["outputs"][0]
        assert "model" not in result["outputs"][0]


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
    return client, fake_dapr


def test_generate_returns_usage_in_metadata(client_and_dapr) -> None:
    """End to end: runtime-reported usage lands in ``LLMChatResponse.metadata``."""
    client, fake_dapr = client_and_dapr
    usage = {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}
    fake_dapr.chat_completion_alpha2.return_value = _raw_response(usage=usage)

    resp = client.generate(messages=[{"role": "user", "content": "hi"}])

    assert resp.metadata["usage"] == usage
    assert isinstance(resp.metadata["usage"]["total_tokens"], int)
