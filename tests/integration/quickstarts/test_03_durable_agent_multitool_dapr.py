"""Integration tests for 03-durable-agent-multitool-dapr quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestDurableAgentMultitoolDaprQuickstart:
    """Integration tests for 03-durable-agent-multitool-dapr quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "03-durable-agent-multitool-dapr"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_multi_tool_agent_dapr(self, dapr_runtime):  # noqa: ARG002
        """Test multi-tool agent example (multi_tool_agent_dapr.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "multi_tool_agent_dapr.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=120,
            use_dapr=True,
            app_id="durablemultitoolapp",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0
