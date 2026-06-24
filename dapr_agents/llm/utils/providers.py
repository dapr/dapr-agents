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

"""Single source of truth for LLM-provider streaming capabilities.

Provider feature lists were previously duplicated across request-building and
stream-processing helpers. Centralize them here so adding a streaming provider
is a one-line change in one place.
"""

from __future__ import annotations

#: Providers with first-class streaming chunk processing (real token deltas).
PROVIDERS_WITH_STREAMING = ("openai", "nvidia")

#: OpenAI-compatible providers that honor ``stream_options={"include_usage": True}``
#: to report token usage on the terminal streaming chunk. Harmless for providers
#: that ignore the flag; only set for those we know accept it.
PROVIDERS_WITH_STREAM_OPTIONS = ("openai", "nvidia")

__all__ = [
    "PROVIDERS_WITH_STREAMING",
    "PROVIDERS_WITH_STREAM_OPTIONS",
]
