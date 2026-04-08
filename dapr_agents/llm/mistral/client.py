import logging
from typing import Any, Optional

try:
    # Mistral SDK v1.x
    from mistralai import Mistral
except ImportError:
    try:
        # Mistral SDK v2.x
        from mistralai.client import Mistral
    except ImportError:
        # Mistral SDK v0.x (Fallback)
        from mistralai.client import MistralClient as Mistral

from dapr_agents.llm.base import LLMClientBase

logger = logging.getLogger(__name__)

class MistralClientBase(LLMClientBase):
    api_key: Optional[str] = None
    endpoint: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        self._provider = "mistral"
        super().model_post_init(__context)
        self.refresh_client()

    def get_config(self) -> Any:
        return {
            "api_key": self.api_key,
            "endpoint": self.endpoint
        }

    def get_client(self) -> Mistral:
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.endpoint:
            kwargs["server_url"] = self.endpoint # v2 uses server_url instead of endpoint

        return Mistral(**kwargs)