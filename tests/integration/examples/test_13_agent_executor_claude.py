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
"""Integration tests for 13-agent-executor-claude example."""

import os

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_script


@pytest.fixture(scope="module")
def anthropic_api_key():
    """Get the Anthropic API key from the environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return api_key


@pytest.mark.integration
class TestAgentExecutorClaudeExample:
    """Integration tests for 13-agent-executor-claude example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir, anthropic_api_key):
        self.example_dir = examples_dir / "13-agent-executor-claude"
        self.env = {"ANTHROPIC_API_KEY": anthropic_api_key, "APPROVAL_MODE": "approve"}

    def test_claude_executor(self, dapr_runtime):  # noqa: ARG002
        """Tool call, session resume and an approved tool call in one session."""
        result = run_quickstart_or_examples_script(
            self.example_dir / "app.py",
            cwd=self.example_dir,
            env=self.env,
            timeout=600,
            use_dapr=True,
            app_id="claude-executor-app",
            resources_path=self.example_dir / "resources",
        )

        assert result.returncode == 0, (
            f"Example failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        for marker in (
            "Turn 1: tool call",
            "Turn 2: resumed session",
            "Approval required",
            "Turn 3: after approval",
        ):
            assert marker in result.stdout, f"missing {marker!r} in output"
        assert "Paris" in result.stdout
