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

from litellm.types.utils import ModelResponseStream

from dapr_agents.llm.litellm.utils import process_litellm_stream
from dapr_agents.types.message import LLMChatResponseChunk


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
