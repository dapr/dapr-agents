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

from litellm.types.utils import ModelResponse, ModelResponseStream

from dapr_agents.llm.litellm.utils import (
    process_litellm_chat_response,
    process_litellm_stream,
)
from dapr_agents.types.message import LLMChatResponse, LLMChatResponseChunk


def test_process_litellm_stream_converts_native_packets():
    stream = iter(
        [
            ModelResponseStream(
                id="litellm-stream",
                created=1,
                model="openai/gpt-4o",
                choices=[
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                    }
                ],
            )
        ]
    )

    chunks = list(process_litellm_stream(stream, on_chunk=None))

    assert len(chunks) == 1
    assert isinstance(chunks[0], LLMChatResponseChunk)
    assert chunks[0].result.content == "Hello"


def test_process_litellm_stream_preserves_metadata_callbacks_and_usage_packets():
    callback_chunks = []
    stream = iter(
        [
            ModelResponseStream(
                id="litellm-stream",
                created=1,
                model="openai/gpt-4o",
                choices=[
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
            ),
            ModelResponseStream(
                id="litellm-stream",
                created=1,
                model="openai/gpt-4o",
                choices=[],
                usage={"total_tokens": 2},
            ),
        ]
    )

    chunks = list(
        process_litellm_stream(
            stream,
            enrich_metadata={"provider": "litellm"},
            on_chunk=callback_chunks.append,
        )
    )

    assert len(chunks) == 2
    assert callback_chunks == [chunks[0]]
    assert chunks[0].metadata["provider"] == "litellm"
    assert chunks[0].metadata["first_chunk"] is True
    assert chunks[0].metadata["last_chunk"] is True
    assert chunks[1].result.content is None
    assert chunks[1].metadata["usage"].total_tokens == 2


def test_process_litellm_chat_response_normalizes_tool_calls_and_metadata():
    response = ModelResponse(
        id="litellm-response",
        created=1,
        model="openai/gpt-4o",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        usage={"total_tokens": 3},
    )

    result = process_litellm_chat_response(response)

    assert isinstance(result, LLMChatResponse)
    assert result.metadata["id"] == "litellm-response"
    assert result.metadata["usage"].total_tokens == 3
    assert result.get_message().tool_calls[0].function.name == "get_weather"
