from pydantic import BaseModel
from typing import Optional, Dict, Any, Union, Iterator, Type, Iterable
from pathlib import Path
from collections import UserDict


class MockLLMClient(UserDict):
    """Mock LLM client for testing."""

    def __init__(self, **kwargs):
        # Initialize UserDict properly
        super().__init__()
        # Set default values
        self.data["model"] = kwargs.get("model", "gpt-4o")
        self.data["azure_deployment"] = kwargs.get("azure_deployment", None)
        self.data["prompt_template"] = kwargs.get("prompt_template", None)
        self.data["api_key"] = kwargs.get("api_key", "mock-api-key")
        self.data["base_url"] = kwargs.get("base_url", "https://api.openai.com/v1")
        self.data["timeout"] = kwargs.get("timeout", 1500)

        # Store additional attributes that might be accessed
        object.__setattr__(
            self, "_prompt_template", kwargs.get("prompt_template", None)
        )
        object.__setattr__(self, "_model", self.data["model"])
        object.__setattr__(self, "_azure_deployment", self.data["azure_deployment"])

    def __getattr__(self, name):
        if name == "data":
            return object.__getattribute__(self, "data")
        if name in self.data:
            return self.data[name]
        return None

    def __setattr__(self, name, value):
        if name == "data" or name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self.data[name] = value

    @classmethod
    def from_prompty(
        cls,
        prompty_source: Union[str, Path],
        timeout: Union[int, float, Dict[str, Any]] = 1500,
    ) -> "MockLLMClient":
        """Mock implementation of from_prompty method."""
        return cls(timeout=timeout)

    def get_client(self):
        """Mock implementation of get_client."""
        return None

    def get_config(self) -> Dict[str, Any]:
        """Mock implementation of get_config."""
        return {
            "model": self.data["model"],
            "azure_deployment": self.data["azure_deployment"],
            "api_key": self.data["api_key"],
            "base_url": self.data["base_url"],
            "timeout": self.data["timeout"],
        }

    def generate(
        self,
        messages: Union[
            str,
            Dict[str, Any],
            Iterable[Union[Dict[str, Any]]],
        ] = None,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[list] = None,
        response_format: Optional[Type[BaseModel]] = None,
        structured_mode: str = "json",
        **kwargs,
    ) -> Union[Iterator[Dict[str, Any]], Dict[str, Any]]:
        """Mock implementation of generate method."""
        return {
            "choices": [{"message": {"content": "Mock response", "role": "assistant"}}]
        }
