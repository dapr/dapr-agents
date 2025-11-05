"""Integration tests for 07-agent-mcp-client-streamablehttp quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestMCPClientStreamableHTTPQuickstart:
    """Integration tests for 07-agent-mcp-client-streamablehttp quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "07-agent-mcp-client-streamablehttp"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_mcp_agent_streamable_http(self, dapr_runtime):  # noqa: ARG002
        """Test MCP agent with streamable HTTP transport (app.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "app.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=120,  # HTTP requires server to be running
            use_dapr=True,
            app_id="mcp-agent-http",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0
