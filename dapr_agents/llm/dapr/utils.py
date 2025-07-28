import logging
import time
from typing import Any, Dict

from dapr_agents.types.message import (
    AssistantMessage,
    LLMChatCandidate,
    LLMChatResponse,
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
    candidates = []
    for choice in response.get("choices", []):
        msg = choice.get("message", {})
        assistant_message = AssistantMessage(
            content=msg.get("content"),
            # Dapr currently never returns refusals, tool_calls or function_call here
        )
        candidate = LLMChatCandidate(
            message=assistant_message,
            finish_reason=choice.get("finish_reason"),
            # Dapr translate_response includes index & no logprobs
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
