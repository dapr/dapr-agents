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

from huggingface_hub import ChatCompletionOutput, ChatCompletionStreamOutput
from huggingface_hub.inference._generated.types.chat_completion import (
    ChatCompletionOutputComplete,
    ChatCompletionOutputMessage,
    ChatCompletionOutputUsage,
    ChatCompletionStreamOutputChoice,
    ChatCompletionStreamOutputDelta,
    ChatCompletionStreamOutputUsage,
)

from dapr_agents.llm.huggingface.utils import (
    process_hf_chat_response,
    process_hf_stream,
)
from dapr_agents.types.message import LLMChatResponseChunk


def _metadata() -> dict:
    return {
        "id": "hf-chat-1",
        "created": 1,
        "model": "test-model",
        "system_fingerprint": "fingerprint",
    }


def test_process_hf_chat_response_uses_typed_attributes() -> None:
    response = ChatCompletionOutput(
        **_metadata(),
        choices=[
            ChatCompletionOutputComplete(
                index=0,
                finish_reason="stop",
                message=ChatCompletionOutputMessage(role="assistant", content="Hello"),
            )
        ],
        usage=ChatCompletionOutputUsage(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        ),
    )

    result = process_hf_chat_response(response)

    assert result.get_message().content == "Hello"
    assert result.metadata["id"] == "hf-chat-1"
    assert result.metadata["usage"].total_tokens == 2


def test_process_hf_chat_response_promotes_tool_call_id() -> None:
    response = ChatCompletionOutput(
        **_metadata(),
        choices=[
            ChatCompletionOutputComplete(
                index=0,
                finish_reason="stop",
                message=ChatCompletionOutputMessage(
                    role="assistant",
                    content=None,
                    tool_call_id="lookup",
                ),
            )
        ],
        usage=ChatCompletionOutputUsage(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        ),
    )

    message = process_hf_chat_response(response).get_message()

    assert message.tool_calls[0].function.name == "lookup"
    assert message.function_call.name == "lookup"


def test_process_hf_stream_uses_typed_attributes_and_usage_packet() -> None:
    raw = [
        ChatCompletionStreamOutput(
            **_metadata(),
            choices=[
                ChatCompletionStreamOutputChoice(
                    index=0,
                    delta=ChatCompletionStreamOutputDelta(
                        role="assistant", content="Hi"
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionStreamOutput(
            **_metadata(),
            choices=[],
            usage=ChatCompletionStreamOutputUsage(
                prompt_tokens=1, completion_tokens=1, total_tokens=2
            ),
        ),
    ]

    chunks = list(process_hf_stream(iter(raw), on_chunk=None))

    assert len(chunks) == 2
    assert all(isinstance(chunk, LLMChatResponseChunk) for chunk in chunks)
    assert chunks[0].result.content == "Hi"
    assert chunks[-1].result.content is None
    assert chunks[-1].metadata["model"] == "test-model"
