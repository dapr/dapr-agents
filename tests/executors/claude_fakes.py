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

"""Offline stand-ins for the Claude Agent SDK client.

Importing this module requires ``claude_agent_sdk``; test modules call
``pytest.importorskip("claude_agent_sdk")`` before importing it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from claude_agent_sdk import ResultMessage

SESSION_ID = "a79daed0-7ae8-4ca5-9385-0d62f44d05d3"


def result_message(**overrides: Any) -> ResultMessage:
    """A successful ``ResultMessage`` with overridable fields."""
    fields: Dict[str, Any] = dict(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id=SESSION_ID,
        total_cost_usd=0.01,
        usage={"input_tokens": 3, "output_tokens": 2},
        result="done",
        stop_reason="end_turn",
        terminal_reason="completed",
    )
    fields.update(overrides)
    return ResultMessage(**fields)


class FakeClaudeClient:
    """Scripted replacement for ``claude_agent_sdk.ClaudeSDKClient``.

    ``script`` holds SDK messages to yield from ``receive_response``; an
    ``Exception`` in the script is raised at that point. ``stderr_lines``
    are fed to the options' stderr callback on connect. Each instance
    records the options it was built with and the prompt it was sent.
    """

    instances: List["FakeClaudeClient"] = []
    script: List[Any] = []
    enter_error: Optional[BaseException] = None
    stderr_lines: List[str] = []

    def __init__(self, options: Any) -> None:
        self.options = options
        self.prompt: Optional[str] = None
        FakeClaudeClient.instances.append(self)

    @classmethod
    def reset(cls, script: Optional[List[Any]] = None) -> None:
        cls.instances = []
        cls.script = list(script or [])
        cls.enter_error = None
        cls.stderr_lines = []

    @classmethod
    def last(cls) -> "FakeClaudeClient":
        return cls.instances[-1]

    async def __aenter__(self) -> "FakeClaudeClient":
        for line in FakeClaudeClient.stderr_lines:
            self.options.stderr(line)
        if FakeClaudeClient.enter_error is not None:
            raise FakeClaudeClient.enter_error
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def query(self, prompt: str) -> None:
        self.prompt = prompt

    async def receive_response(self):
        for message in list(FakeClaudeClient.script):
            if isinstance(message, BaseException):
                raise message
            yield message
