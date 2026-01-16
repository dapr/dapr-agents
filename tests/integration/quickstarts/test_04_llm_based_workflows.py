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
        script = self.quickstart_dir / "01_single_activity_workflow.py"
        result = run_quickstart_script(
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
        result = run_quickstart_script(
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
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_parallel_workflow(self, dapr_runtime):  # noqa: ARG002
        script = self.quickstart_dir / "04_parallel_workflow.py"
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
