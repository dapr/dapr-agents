"""Integration tests for 04-llm-based-workflows quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestLLMBasedWorkflowsQuickstart:
    """Integration tests for 04-llm-based-workflows quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "04-llm-based-workflows"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_single_activity_workflow(self, dapr_runtime):  # noqa: ARG002
        """Test single activity workflow (01_single_activity_workflow.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "01_single_activity_workflow.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=120,  # Workflows with Dapr may take longer
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
        """Test sequential workflow (03_sequential_workflow.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "03_sequential_workflow.py"
        result = run_quickstart_script(
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
