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

"""Helpers for identifying persisted Drasi subscriptions."""

DRASI_SUBSCRIPTION_INSTRUCTIONS_KEY_PREFIX = "drasi_subscriptions:"
# TODO: remove hardcoded agent ID in favor of having tool hook context/runtime provide it
TEST_AGENT_ID = "InventoryAgent"


def build_subscription_key(
    query_id: str,
    agent_id: str,
    subscription_id: str,
) -> str:
    """Build the state key shared by Drasi subscription hooks and tools."""
    identifier = f"{query_id}:{agent_id}:{subscription_id}"
    return f"{DRASI_SUBSCRIPTION_INSTRUCTIONS_KEY_PREFIX}{identifier}"
