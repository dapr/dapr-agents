"""Integration tests for 05-multi-agent-workflows quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_multi_app


@pytest.mark.integration
class TestMultiAgentWorkflowsQuickstart:
    """Integration tests for 05-multi-agent-workflows quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "05-multi-agent-workflows"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_random_orchestrator(self, dapr_runtime):  # noqa: ARG002
        dapr_yaml = self.quickstart_dir / "dapr-random.yaml"
        result = run_quickstart_multi_app(
            dapr_yaml,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=300,
            stream_logs=True,  # Stream logs in real-time for debugging
        )

        assert result.returncode == 0, (
            f"Multi-app run failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    # def test_roundrobin_orchestrator(self, dapr_runtime):  # noqa: ARG002
    #     dapr_yaml = self.quickstart_dir / "dapr-roundrobin.yaml"
    #     result = run_quickstart_multi_app(
    #         dapr_yaml,
    #         cwd=self.quickstart_dir,
    #         env=self.env,
    #         timeout=300,
    #     )

    #     assert result.returncode == 0, (
    #         f"Multi-app run failed with return code {result.returncode}.\n"
    #         f"STDOUT:\n{result.stdout}\n"
    #         f"STDERR:\n{result.stderr}"
    #     )
    #     # expect some output
    #     assert len(result.stdout) > 0 or len(result.stderr) > 0

    # def test_llm_orchestrator(self, dapr_runtime):  # noqa: ARG002
    #     dapr_yaml = self.quickstart_dir / "dapr-llm.yaml"
    #     result = run_quickstart_multi_app(
    #         dapr_yaml,
    #         cwd=self.quickstart_dir,
    #         env=self.env,
    #         timeout=300,
    #     )

    #     assert result.returncode == 0, (
    #         f"Multi-app run failed with return code {result.returncode}.\n"
    #         f"STDOUT:\n{result.stdout}\n"
    #         f"STDERR:\n{result.stderr}"
    #     )
    #     # expect some output
    #     assert len(result.stdout) > 0 or len(result.stderr) > 0
