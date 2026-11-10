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

"""Integration test for the workflow event routes quickstart against a real daprd.

The script starts a workflow, publishes a message through the sidecar and the
WorkflowEventRouteSpec raises the external event on that workflow. No LLM is
involved, so this test needs neither an API key nor Ollama.
"""

import pytest

from tests.integration.quickstarts.conftest import run_quickstart_or_examples_script


@pytest.mark.integration
class TestWorkflowEventRoutesQuickstart:
    """Pub/sub message -> WorkflowEventRouteSpec -> raise_workflow_event on daprd."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir):
        self.quickstart_dir = quickstarts_dir

    def test_10_workflow_event_routes(self, dapr_runtime):  # noqa: ARG002
        script_path = self.quickstart_dir / "10_workflow_event_routes.py"
        result = run_quickstart_or_examples_script(
            script_path,
            cwd=self.quickstart_dir,
            timeout=180,
            use_dapr=True,
            app_id="workflow-event-routes",
            resources_path=self.quickstart_dir / "resources",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert "finished with COMPLETED" in result.stdout
        assert "'status': 'done'" in result.stdout
        assert "Event route delivered the event." in result.stdout
