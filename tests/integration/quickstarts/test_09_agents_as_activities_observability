"""Integration tests for 09-agents-as-activities-observability quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestAgentsAsTasksInWorkflowsQuickstart:
    """Integration tests for 09-agents-as-activities-observability quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "09-agents-as-activities-observability"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_sequential_workflow(self, dapr_runtime):  # noqa: ARG002
        """Test sequential workflow with agents as tasks (sequential_workflow.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "sequential_workflow.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # Agent workflows may take longer
            use_dapr=True,
            app_id="dapr-agent-wf",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert "Workflow started:" in result.stdout or len(result.stdout) > 0

    def test_sequential_workflow_tracing(self):
        pytest.skip(
            "Skipping tracing test (test_sequential_workflow_tracing) in CI for now"
        )

    def test_sequential_workflow_multi_model_tracing(self):
        pytest.skip(
            "Skipping tracing test (test_sequential_workflow_multi_model_tracing) in CI for now"
        )
