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

"""Regression tests for OpenAI streaming chunk normalization.

Focus: the final usage-only packet OpenAI emits when
``stream_options.include_usage`` is enabled carries an empty ``choices`` list.
That packet must still produce a valid ``LLMChatResponseChunk`` (whose
``result`` field is required) rather than raising a Pydantic validation error
inside the workflow's ``call_llm`` activity.
"""

from unittest.mock import MagicMock

from dapr_agents.llm.openai.utils import process_openai_stream
from dapr_agents.types.message import LLMChatResponseChunk


def _packet(data: dict) -> MagicMock:
    """Wrap a dict as an OpenAI SDK chunk exposing ``model_dump``."""
    pkt = MagicMock()
    pkt.model_dump.return_value = data
    return pkt


def _content_packet(content: str, *, finish_reason=None, role=None) -> MagicMock:
    delta: dict = {"content": content}
    if role:
        delta["role"] = role
    return _packet(
        {
            "id": "chatcmpl-1",
            "created": 1,
            "model": "gpt-4o-mini",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


def _usage_only_packet() -> MagicMock:
    # OpenAI's terminal packet when include_usage is on: empty choices + usage.
    return _packet(
        {
            "id": "chatcmpl-1",
            "created": 1,
            "model": "gpt-4o-mini",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "completion_tokens": 238,
                "prompt_tokens": 638,
                "total_tokens": 876,
            },
        }
    )


def test_usage_only_final_packet_yields_valid_chunk():
    """The empty-choices usage packet must not crash chunk construction."""
    raw = [
        _content_packet("Hello", role="assistant"),
        _content_packet(" world", finish_reason="stop"),
        _usage_only_packet(),
    ]

    chunks = list(process_openai_stream(iter(raw), on_chunk=None))

    # Every yielded item is a valid response chunk with a populated result.
    assert len(chunks) == 3
    assert all(isinstance(c, LLMChatResponseChunk) for c in chunks)
    assert all(c.result is not None for c in chunks)

    # The terminal usage packet carries an empty candidate (no content/finish)
    # but preserves provider metadata for downstream TURN_COMPLETE attribution.
    final = chunks[-1]
    assert final.result.content is None
    assert final.result.finish_reason is None
    assert final.metadata is not None
    assert final.metadata.get("model") == "gpt-4o-mini"


def test_content_chunks_reconstruct_full_message():
    """Content deltas accumulate correctly alongside the usage-only tail."""
    raw = [
        _content_packet("Hel", role="assistant"),
        _content_packet("lo", finish_reason="stop"),
        _usage_only_packet(),
    ]

    chunks = list(process_openai_stream(iter(raw), on_chunk=None))
    text = "".join(c.result.content or "" for c in chunks)
    assert text == "Hello"
