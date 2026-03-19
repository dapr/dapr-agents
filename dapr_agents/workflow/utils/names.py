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

import re


def sanitize_agent_name(name: str) -> str:
    """
    Sanitize an agent name for use in Dapr workflow IDs.

    Keeps only alphanumeric characters, hyphens, and underscores.
    All other characters are replaced with underscores.

    Args:
        name: The agent name to sanitize.

    Returns:
        A sanitized name safe for use in Dapr workflow IDs.

    Example:
        >>> sanitize_agent_name("my-agent@123")
        'my-agent_123'
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)
