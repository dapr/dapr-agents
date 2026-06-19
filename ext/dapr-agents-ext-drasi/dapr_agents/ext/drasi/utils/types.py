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

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


# Public-facing operation type
DrasiOperation = Literal["i", "u", "d", "x"]


# TODO: replace these with exported Drasi models from the reaction SDK
class DrasiUnpackedSource(BaseModel):
    """Source information for a Drasi event."""

    queryId: str = Field(description="The query ID that generated this event")
    ts_ms: int = Field(description="Source timestamp in milliseconds")


class DrasiUnpackedPayload(BaseModel):
    """Payload containing the event data."""

    source: DrasiUnpackedSource = Field(description="Source information")
    after: Optional[Dict[str, Any]] = Field(
        default=None, description="Record state after the change"
    )
    before: Optional[Dict[str, Any]] = Field(
        default=None, description="Record state before the change"
    )


class DrasiUnpackedEvent(BaseModel):
    """Drasi unpacked event model for CDC (Change Data Capture) events."""

    op: DrasiOperation = Field(
        description="Operation type: i (insert), u (update), d (delete), x (control)"
    )
    ts_ms: int = Field(description="Event timestamp in milliseconds")
    seq: int = Field(description="Event sequence number")
    payload: DrasiUnpackedPayload = Field(
        description="Event payload containing source and data"
    )
