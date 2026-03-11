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

"""Integration tests for 01-dapr-agents-fundamentals quickstart."""

import os

import pytest
from tests.integration.quickstarts.conftest import (
    run_quickstart_or_examples_multi_app,
    run_quickstart_or_examples_script,
)


@pytest.mark.integration
class TestHelloWorldQuickstart:
    """Integration tests for 01-dapr-agents-fundamentals quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key, ollama_resources_dir, is_ollama):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir
        self.env = {"OPENAI_API_KEY": openai_api_key}
        self.resources_path = ollama_resources_dir
        self.is_ollama = is_ollama
        if is_ollama:
            self.env["OLLAMA_MODEL"] = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

    @pytest.mark.ollama
    def test_01_llm_client(self, dapr_runtime):  # noqa: ARG002
        """Test LLM client example (01_llm_client.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "01_llm_client.py"
        result = run_quickstart_or_examples_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="llm-client",
            resources_path=self.resources_path,
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
        # Verify LLM response was received
        assert "Response:" in result.stdout

    @pytest.mark.ollama
    def test_02_durable_agent_http(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent HTTP example (02_durable_agent_http.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "02_durable_agent_http.py"
        result = run_quickstart_or_examples_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="durable-agent",
            app_port=8001,
            resources_path=self.resources_path,
            trigger_curl={
                "url": "http://localhost:8001/agent/run",
                "method": "POST",
                "data": {"task": "What is the weather in London?"},
                "headers": {"Content-Type": "application/json"},
                "wait_seconds": 5,
            },
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_03_durable_agent_pubsub(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent pub/sub example (03_durable_agent_pubsub.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "03_durable_agent_pubsub.py"
        result = run_quickstart_or_examples_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="durable-agent-sub",
            dapr_http_port=3500,
            resources_path=self.resources_path,
            trigger_pubsub={
                "pubsub_name": "agent-pubsub",
                "topic": "weather.requests",
                "data": {"task": "What's the weather in Boston?"},
                "wait_seconds": 5,
            },
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    @pytest.mark.ollama
    def test_04_workflow_llm(self, dapr_runtime):  # noqa: ARG002
        """Test workflow with LLM activities example (04_workflow_llm.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "04_workflow_llm.py"
        result = run_quickstart_or_examples_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="workflow-llms",
            resources_path=self.resources_path,
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
        # Verify workflow completed successfully
        assert (
            "Workflow started:" in result.stdout
            or "Blog post" in result.stdout
            or "Final Blog Post" in result.stdout
        )

    def test_05_workflow_agents(self, dapr_runtime):  # noqa: ARG002
        """Test workflow with agent activities example (05_workflow_agents.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        dapr_yaml = self.quickstart_dir / "05_workflow_agents.yaml"
        result = run_quickstart_or_examples_multi_app(
            dapr_yaml,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            stream_logs=True,
        )

        assert result.returncode == 0, (
            f"Multi-app quickstart '{dapr_yaml}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        combined = (result.stdout or "") + (result.stderr or "")
        assert (
            "Workflow started:" in combined
            or "Recommendation" in combined
            or "Final Recommendation" in combined
        ), (
            f"Expected workflow output in combined stdout+stderr. Got: {combined[:500]!r}..."
        )

    def test_06_durable_agent_tracing(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent tracing example (06_durable_agent_tracing.py)."""
        script_path = self.quickstart_dir / "06_durable_agent_tracing.py"
        result = run_quickstart_or_examples_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="durable-agent-trace",
            resources_path=self.resources_path,
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
