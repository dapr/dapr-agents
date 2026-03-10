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

"""Integration tests for 06-agent-mcp-client-stdio example."""

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_script


@pytest.mark.integration
class TestMCPClientStdioQuickstart:
    """Integration tests for 06-agent-mcp-client-stdio example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = examples_dir / "06-agent-mcp-client-stdio"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_mcp_agent_stdio(self, dapr_runtime):  # noqa: ARG002
        """Test MCP agent with STDIO transport (agent.py)."""
        script = self.quickstart_dir / "agent.py"
        result = run_quickstart_or_examples_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="mcp-agent-stdio",
            stream_logs=True,
            trigger_curl={
                "url": "http://localhost:8001/agent/run",
                "method": "POST",
                "data": {"task": "What is the weather in New York?"},
                "headers": {"Content-Type": "application/json"},
                "wait_seconds": 10,
            },
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert "Available tools" in result.stdout or len(result.stdout) > 0
