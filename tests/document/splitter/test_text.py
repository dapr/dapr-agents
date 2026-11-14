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

from dapr_agents.document.splitter.text import TextSplitter
from dapr_agents.types.document import Document


def test_merge_splits_does_not_drop_content_that_triggers_finalization():
    """A split that overflows the current chunk must still end up in the
    output somewhere, not be silently discarded when the in-progress chunk
    is finalized to make room for it."""
    splitter = TextSplitter(
        chunk_size=10150, chunk_overlap=0, separator=None, fallback_separators=[]
    )
    splits = ["A" * 100, "B" * 100, "C" * 3000]

    merged = splitter._merge_splits(splits, max_size=150)

    combined = "".join(merged)
    assert combined.count("A") == 100
    assert combined.count("B") == 100
    assert combined.count("C") == 3000


def test_merge_splits_respects_max_size_for_normal_splits():
    splitter = TextSplitter(
        chunk_size=10150, chunk_overlap=0, separator=None, fallback_separators=[]
    )
    splits = ["A" * 50, "B" * 50, "C" * 50, "D" * 50]

    merged = splitter._merge_splits(splits, max_size=100)

    assert merged == ["A" * 50 + "B" * 50, "C" * 50 + "D" * 50]


def test_merge_splits_with_overlap_carries_previous_content_forward():
    splitter = TextSplitter(
        chunk_size=10150, chunk_overlap=10, separator=None, fallback_separators=[]
    )
    splits = ["A" * 50, "B" * 50, "C" * 50]

    merged = splitter._merge_splits(splits, max_size=100)

    assert len(merged) >= 2
    # every split must appear in at least one chunk; none dropped
    combined = "".join(merged)
    assert combined.count("A") == 50
    assert combined.count("B") == 50
    assert combined.count("C") == 50


def test_split_short_text_returns_single_chunk():
    splitter = TextSplitter(chunk_size=100, chunk_overlap=0)
    text = "short text"

    assert splitter.split(text) == [text]


def test_split_long_text_preserves_all_content():
    splitter = TextSplitter(chunk_size=50, chunk_overlap=0, separator="\n\n")
    text = "\n\n".join(f"paragraph {i} " + "x" * 40 for i in range(10))

    chunks = splitter.split(text)

    for i in range(10):
        assert any(f"paragraph {i} " in chunk for chunk in chunks), (
            f"paragraph {i} missing from output chunks"
        )


def test_split_documents_preserves_source_offsets_with_overlap():
    source = "alpha beta gamma"
    splitter = TextSplitter(chunk_size=11, chunk_overlap=5)
    chunks = splitter.split_documents([Document(text=source)])

    assert len(chunks) == 2
    for chunk in chunks:
        start = chunk.metadata["start_index"]
        end = chunk.metadata["end_index"]
        assert source[start:end] == chunk.text

    assert chunks[0].text == "alpha beta "
    assert (chunks[0].metadata["start_index"], chunks[0].metadata["end_index"]) == (
        0,
        11,
    )
    assert chunks[1].text == "beta gamma"
    assert (chunks[1].metadata["start_index"], chunks[1].metadata["end_index"]) == (
        6,
        16,
    )


def test_split_documents_preserves_character_offsets_with_custom_sizing_function():
    source = "alpha beta gamma delta"
    splitter = TextSplitter(
        chunk_size=2,
        chunk_overlap=0,
        chunk_size_function=lambda s: len(s.split()),
    )
    chunks = splitter.split_documents([Document(text=source)])

    assert len(chunks) == 2
    for chunk in chunks:
        start = chunk.metadata["start_index"]
        end = chunk.metadata["end_index"]
        assert source[start:end] == chunk.text
        assert chunk.metadata["chunk_length"] == 2

    assert (chunks[0].metadata["start_index"], chunks[0].metadata["end_index"]) == (
        0,
        11,
    )
    assert (chunks[1].metadata["start_index"], chunks[1].metadata["end_index"]) == (
        11,
        22,
    )


def test_split_documents_preserves_offsets_with_repeated_phrases():
    source = "item item item item"
    splitter = TextSplitter(chunk_size=10, chunk_overlap=4, separator=" ")
    chunks = splitter.split_documents([Document(text=source)])

    assert len(chunks) > 1
    last_start = -1
    for chunk in chunks:
        start = chunk.metadata["start_index"]
        end = chunk.metadata["end_index"]
        assert start > last_start
        assert source[start:end] == chunk.text
        last_start = start
