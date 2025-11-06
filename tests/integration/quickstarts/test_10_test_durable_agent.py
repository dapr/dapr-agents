"""Integration tests for 10-test-durable-agent quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestDurableAgentQuickstart:
    """Integration tests for 10-test-durable-agent quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "10-test-durable-agent"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_durable_agent_app(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent app (app.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "app.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=120,
            use_dapr=True,
            app_id="blog-app-agent",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0
