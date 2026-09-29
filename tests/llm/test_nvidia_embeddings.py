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

from unittest.mock import MagicMock
from dapr_agents.llm.nvidia.embeddings import NVIDIAEmbeddingClient


def _client_with_captured_body(**client_kwargs):
    client = NVIDIAEmbeddingClient(api_key="fake", **client_kwargs)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    client.client.embeddings.create = fake_create
    return client, captured


def test_create_embedding_uses_instance_input_type_and_truncate_by_default():
    client, captured = _client_with_captured_body(input_type="query", truncate="END")
    client.create_embedding(input="hello")

    assert captured["extra_body"]["input_type"] == "query"
    assert captured["extra_body"]["truncate"] == "END"


def test_create_embedding_call_args_override_instance_defaults():
    client, captured = _client_with_captured_body(input_type="query", truncate="END")
    client.create_embedding(input="hello", input_type="passage", truncate="START")

    assert captured["extra_body"]["input_type"] == "passage"
    assert captured["extra_body"]["truncate"] == "START"
