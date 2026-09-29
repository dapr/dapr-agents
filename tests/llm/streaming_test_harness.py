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

"""Reusable compliance harness for verifying LLM streaming implementations.

Validates that provider-specific stream processors and the centralized
`StreamHandler.process_stream` conform to the canonical `LLMChatResponseChunk`
streaming specification.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from dapr_agents.types.message import LLMChatCandidateChunk, LLMChatResponseChunk
from dapr_agents.types.streaming import AssistantMessageAccumulator


class StreamingComplianceHarness:
    """Reusable assertions and test helpers for LLM provider streaming pipelines."""

    @staticmethod
    def assert_valid_stream_contract(
        chunks: Sequence[LLMChatResponseChunk],
        expected_provider: Optional[str] = None,
        expected_min_chunks: int = 1,
    ) -> None:
        """Validate envelope and typing invariants across all stream chunks.

        - Every item must be an instance of LLMChatResponseChunk.
        - Every chunk must have a non-null result of type LLMChatCandidateChunk.
        - Every chunk must have a metadata dictionary.
        - If expected_provider is set, metadata["provider"] must match.
        """
        assert len(chunks) >= expected_min_chunks, (
            f"Expected at least {expected_min_chunks} chunks, got {len(chunks)}"
        )
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, LLMChatResponseChunk), (
                f"Chunk {i} is not an LLMChatResponseChunk (type: {type(chunk)})"
            )
            assert isinstance(chunk.result, LLMChatCandidateChunk), (
                f"Chunk {i}.result is not an LLMChatCandidateChunk (type: {type(chunk.result)})"
            )
            assert isinstance(chunk.metadata, dict), (
                f"Chunk {i}.metadata is not a dict (type: {type(chunk.metadata)})"
            )
            if expected_provider is not None:
                actual_provider = chunk.metadata.get("provider")
                assert actual_provider == expected_provider, (
                    f"Chunk {i} provider mismatch: expected '{expected_provider}', got '{actual_provider}'"
                )

    @staticmethod
    def assert_reconstructs_text(
        chunks: Sequence[LLMChatResponseChunk],
        expected_text: str,
        expected_finish_reason: Optional[str] = "stop",
    ) -> None:
        """Validate that text content deltas fold into the expected assistant message."""
        accumulator = AssistantMessageAccumulator()
        for chunk in chunks:
            accumulator.ingest(chunk)

        msg = accumulator.assistant_message()
        actual_content = msg.get("content")
        assert actual_content == expected_text, (
            f"Reconstructed text mismatch: expected '{expected_text}', got '{actual_content}'"
        )
        if expected_finish_reason is not None:
            assert accumulator.finish_reason == expected_finish_reason, (
                f"Finish reason mismatch: expected '{expected_finish_reason}', got '{accumulator.finish_reason}'"
            )

    @staticmethod
    def assert_reconstructs_tool_calls(
        chunks: Sequence[LLMChatResponseChunk],
        expected_tools: Sequence[Dict[str, Any]],
        expected_finish_reason: Optional[str] = "tool_calls",
    ) -> None:
        """Validate that streaming tool call deltas reconstruct into full tool calls."""
        accumulator = AssistantMessageAccumulator()
        for chunk in chunks:
            accumulator.ingest(chunk)

        msg = accumulator.assistant_message()
        tool_calls = msg.get("tool_calls")
        assert tool_calls is not None, "Reconstructed message contains no tool_calls"
        assert len(tool_calls) == len(expected_tools), (
            f"Expected {len(expected_tools)} tool call(s), got {len(tool_calls)}"
        )

        for i, (actual, expected) in enumerate(zip(tool_calls, expected_tools)):
            if "name" in expected:
                assert actual.get("function", {}).get("name") == expected["name"], (
                    f"Tool {i} name mismatch: expected '{expected['name']}', got '{actual.get('function', {}).get('name')}'"
                )
            if "arguments" in expected:
                actual_args = actual.get("function", {}).get("arguments", "")
                expected_args = expected["arguments"]
                if isinstance(expected_args, dict):
                    assert json.loads(actual_args) == expected_args
                else:
                    assert expected_args in actual_args

        if expected_finish_reason is not None:
            assert accumulator.finish_reason == expected_finish_reason, (
                f"Finish reason mismatch: expected '{expected_finish_reason}', got '{accumulator.finish_reason}'"
            )

    @staticmethod
    def assert_on_chunk_callback_lifecycle(
        stream_factory: Callable[
            [Callable[[LLMChatResponseChunk], None]], Iterator[LLMChatResponseChunk]
        ],
    ) -> List[LLMChatResponseChunk]:
        """Verify on_chunk callback receives every yielded chunk in identical order and identity."""
        captured: List[LLMChatResponseChunk] = []
        yielded = list(stream_factory(captured.append))

        assert len(captured) == len(yielded), (
            f"Callback count mismatch: on_chunk fired {len(captured)} times, yielded {len(yielded)} chunks"
        )
        for i, (c, y) in enumerate(zip(captured, yielded)):
            assert c is y, (
                f"Chunk {i} passed to on_chunk is not identical to yielded chunk"
            )

        return yielded

    @staticmethod
    def assert_snapshot_isolation(
        chunks: Sequence[LLMChatResponseChunk],
    ) -> None:
        """Verify that buffered chunks do not share mutable metadata dictionaries."""
        if len(chunks) < 2:
            return

        for i in range(len(chunks) - 1):
            assert chunks[i].metadata is not chunks[i + 1].metadata, (
                f"Chunk {i} and {i + 1} share the same metadata dict reference (missing snapshot isolation)"
            )
