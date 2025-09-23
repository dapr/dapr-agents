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
