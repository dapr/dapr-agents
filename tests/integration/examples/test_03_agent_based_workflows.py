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

"""Integration tests for 03-agent-based-workflows example."""

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_multi_app


@pytest.mark.integration
class TestAgentBasedWorkflowsQuickstart:
    """Integration tests for 03-agent-based-workflows example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = examples_dir / "03-agent-based-workflows"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_sequential_workflow(self, dapr_runtime):  # noqa: ARG002
        """Test sequential agent workflow (01_sequential_workflow.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        dapr_yaml = self.quickstart_dir / "sequential.yaml"
        # TODO: rm this once we get the next Dapr CLI RC with this set by default.
        env = self.env.copy()
        env["DAPR_HOST_IP"] = "127.0.0.1"
        result = run_quickstart_or_examples_multi_app(
            dapr_yaml,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # Agent workflows may take longer
            stream_logs=True,
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert "Workflow started:" in result.stdout or "Itinerary" in result.stdout
