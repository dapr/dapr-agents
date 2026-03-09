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
import time
from typing import Any, Dict, Optional, List

from dapr_agents.types.message import (
    AssistantMessage,
    LLMChatCandidate,
    LLMChatResponse,
    ToolCall,
    FunctionCall,
)

logger = logging.getLogger(__name__)


def process_dapr_chat_response(response: Dict[str, Any]) -> LLMChatResponse:
    """
    Convert a Dapr-normalized chat dict (with OpenAI-style 'choices') into a unified LLMChatResponse.

    Args:
        response: The dict returned by `DaprChatClient.translate_response`.

    Returns:
        LLMChatResponse: Contains a list of candidates and metadata.
    """
    # 1) Extract each choice → build AssistantMessage + LLMChatCandidate
    candidates: List[LLMChatCandidate] = []
    for choice in response.get("choices", []):
        msg = choice.get("message", {})

        # Build tool_calls if present (OpenAI-like)
        tool_calls: Optional[List[ToolCall]] = None
        if msg.get("tool_calls"):
            tool_calls = []
            for tc in msg["tool_calls"]:
                try:
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            type=tc.get("type", "function"),
                            function=FunctionCall(
                                name=tc.get("function", {}).get("name", ""),
                                arguments=tc.get("function", {}).get("arguments", ""),
                            ),
                        )
                    )
                except Exception:
                    logger.exception(f"Invalid tool_call entry: {tc}")

        function_call = (
            None  # there is no openai "function_call" in dapr only tool calls
        )

        content = msg.get("content")
        if isinstance(content, dict):
            try:
                import json

                content = json.dumps(content)
            except Exception as e:
                logger.warning(f"Failed to serialize dictionary content: {e}")

        assistant_message = AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            function_call=function_call,
        )
        candidate = LLMChatCandidate(
            message=assistant_message,
            finish_reason=choice.get("finish_reason"),
            index=choice.get("index"),
            logprobs=choice.get("logprobs"),
        )
        candidates.append(candidate)

    # 2) Build metadata from the top‐level fields
    metadata: Dict[str, Any] = {
        "provider": "dapr",
        "id": response.get("id", None),
        "model": response.get("model", None),
        "object": response.get("object", None),
        "usage": response.get("usage", {}),
        "created": response.get("created", int(time.time())),
    }

    return LLMChatResponse(results=candidates, metadata=metadata)
