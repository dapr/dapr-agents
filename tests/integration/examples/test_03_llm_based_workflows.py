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

"""Integration tests for 03-llm-based-workflows example."""

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_script


@pytest.mark.integration
class TestLLMBasedWorkflowsQuickstart:
    """Integration tests for 03-llm-based-workflows example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir, openai_api_key, is_ollama):
        """Setup test environment."""
        self.quickstart_dir = examples_dir / "03-llm-based-workflows"
        self.env = {"OPENAI_API_KEY": openai_api_key}
        if is_ollama:
            import os

            self.env["OPENAI_MODEL"] = os.environ["OLLAMA_MODEL"]
            self.env["OPENAI_BASE_URL"] = os.environ["OLLAMA_ENDPOINT"]

    def test_single_activity_workflow(self, dapr_runtime):  # noqa: ARG002
        script = self.quickstart_dir / "01_single_activity_workflow.py"
        result = run_quickstart_or_examples_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # Workflows with Dapr may take longer
            use_dapr=True,
            app_id="dapr-agent-wf-sequence",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert "Workflow started:" in result.stdout or "bio" in result.stdout.lower()

    def test_single_structured_activity_workflow(self, dapr_runtime):  # noqa: ARG002
        script = self.quickstart_dir / "02_single_structured_activity_workflow.py"
        result = run_quickstart_or_examples_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # Workflows with Dapr may take longer
            use_dapr=True,
            app_id="dapr-agent-wf-sequence",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert "Workflow started:" in result.stdout or "bio" in result.stdout.lower()

    def test_sequential_workflow(self, dapr_runtime):  # noqa: ARG002
        script = self.quickstart_dir / "03_sequential_workflow.py"
        result = run_quickstart_or_examples_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # Sequential workflows may take longer
            use_dapr=True,
            app_id="dapr-agent-wf-sequence",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_parallel_workflow(self, dapr_runtime):  # noqa: ARG002
        script = self.quickstart_dir / "04_parallel_workflow.py"
        result = run_quickstart_or_examples_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # Sequential workflows may take longer
            use_dapr=True,
            app_id="dapr-agent-wf-sequence",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
