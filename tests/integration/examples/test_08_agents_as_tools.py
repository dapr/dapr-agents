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

"""Integration tests for 08-agents-as-tools example."""

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_multi_app


@pytest.mark.integration
class TestAgentsAsToolsExample:
    """Integration tests for 08-agents-as-tools example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir, openai_api_key, is_ollama):
        self.example_dir = examples_dir / "08-agents-as-tools"
        self.env = {"OPENAI_API_KEY": openai_api_key}
        if is_ollama:
            import os

            self.env["OPENAI_MODEL"] = os.environ["OLLAMA_MODEL"]
            self.env["OPENAI_BASE_URL"] = os.environ["OLLAMA_ENDPOINT"]

    def test_cross_app(self, dapr_runtime):  # noqa: ARG002
        """Sam runs in a separate Dapr app; Frodo discovers and calls it as a tool."""
        dapr_yaml = self.example_dir / "dapr-cross-app.yaml"
        result = run_quickstart_or_examples_multi_app(
            dapr_yaml,
            cwd=self.example_dir,
            env={**self.env, "DAPR_HOST_IP": "127.0.0.1"},
            timeout=300,
            trigger_curl={
                "url": "http://localhost:8001/agent/run",
                "method": "POST",
                "data": {
                    "task": "What supplies do we have for the next leg of the journey? Ask Sam."
                },
                "headers": {"Content-Type": "application/json"},
                "app_port": 8001,
                "wait_seconds": 30,
            },
        )

        assert result.returncode == 0, (
            f"Cross-app agents-as-tools failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
