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

"""Tests for the streaming chunk envelope and types."""

from __future__ import annotations

import json

import pytest

from dapr_agents.types.streaming import (
    STREAM_SCHEMA_VERSION,
    AgentStreamChunk,
    StreamChunkType,
    StreamDelta,
    UserInputResponse,
)


def _minimal_chunk(**overrides) -> AgentStreamChunk:
    base = {
        "sequence": 1,
        "type": StreamChunkType.CONTENT_DELTA,
        "agent": "alice",
        "workflow_instance_id": "wf-1",
        "turn": 0,
        "root_instance_id": "root-1",
    }
    base.update(overrides)
    return AgentStreamChunk(**base)


class TestAgentStreamChunk:
    def test_defaults(self) -> None:
        chunk = _minimal_chunk()
        assert chunk.schema_version == STREAM_SCHEMA_VERSION
        assert chunk.chunk_id  # auto-generated
        assert chunk.parent_agent is None
        assert chunk.depth == 0
        assert chunk.call_path == []
        assert chunk.complete_message is None
        assert chunk.delta is None

    def test_json_roundtrip(self) -> None:
        chunk = _minimal_chunk(
            delta=StreamDelta(content="hello"),
            call_path=["alice", "bob"],
            depth=1,
        )
        payload = chunk.model_dump_json()
        loaded = AgentStreamChunk.model_validate_json(payload)
        assert loaded == chunk

    def test_unique_chunk_ids(self) -> None:
        c1 = _minimal_chunk()
        c2 = _minimal_chunk()
        assert c1.chunk_id != c2.chunk_id

    def test_forbid_extra(self) -> None:
        with pytest.raises(Exception):
            AgentStreamChunk.model_validate(
                {
                    "sequence": 1,
                    "type": "content_delta",
                    "agent": "a",
                    "workflow_instance_id": "wf",
                    "turn": 0,
                    "root_instance_id": "r",
                    "unknown_field": "nope",
                }
            )

    def test_all_chunk_types_serialize(self) -> None:
        for t in StreamChunkType:
            chunk = _minimal_chunk(type=t)
            assert chunk.type is t
            loaded = AgentStreamChunk.model_validate_json(chunk.model_dump_json())
            assert loaded.type is t

    def test_timestamp_is_iso8601(self) -> None:
        from datetime import datetime

        chunk = _minimal_chunk()
        assert "T" in chunk.timestamp
        # Python ``datetime.isoformat()`` on a UTC-aware datetime produces
        # ``+00:00``, never ``Z``. The old assertion ``or .endswith("Z")``
        # was a dead branch giving false confidence.
        assert chunk.timestamp.endswith("+00:00")
        # Parse round-trip: verify it's genuinely ISO 8601, not just a
        # coincidental suffix match.
        parsed = datetime.fromisoformat(chunk.timestamp)
        assert parsed.utcoffset() is not None
        assert parsed.utcoffset().total_seconds() == 0


class TestStreamDelta:
    def test_all_optional(self) -> None:
        d = StreamDelta()
        assert d.content is None
        assert d.tool_calls is None

    def test_forbid_extra(self) -> None:
        with pytest.raises(Exception):
            StreamDelta.model_validate({"content": "x", "bogus": 1})


class TestUserInputResponse:
    def test_required_fields(self) -> None:
        resp = UserInputResponse(
            request_id="r1",
            target_instance_id="wf-x",
            answer="yes",
        )
        assert resp.model_dump() == {
            "request_id": "r1",
            "target_instance_id": "wf-x",
            "answer": "yes",
        }

    def test_rejects_missing_field(self) -> None:
        with pytest.raises(Exception):
            UserInputResponse(request_id="r1", answer="y")
