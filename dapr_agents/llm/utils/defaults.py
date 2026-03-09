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

import logging
from typing import Optional

from dapr_agents.llm.chat import ChatClientBase

logger = logging.getLogger(__name__)


def get_default_llm() -> Optional[ChatClientBase]:
    """
    Centralized default LLM factory for the SDK.

    Returns:
        Optional[ChatClientBase]: A configured default LLM client or None if not available.
    """
    try:
        from dapr_agents.llm.dapr import DaprChatClient

        return DaprChatClient()
    except Exception as e:
        logger.warning(f"Failed to create default Dapr client: {e}. LLM will be None.")
        raise
