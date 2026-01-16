"""Integration tests for 07-agent-mcp-client-streamablehttp quickstart."""
import pytest
from tests.integration.quickstarts.conftest import (
    run_quickstart_script,
    MCPServerContext,
)


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

        This test starts the MCP server first, then runs the agent.
        The MCP server must be running on port 8000 for the agent to connect.

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        """
        with MCPServerContext(
            self.quickstart_dir,
            server_type="streamable-http",
            port=8000,
            env=self.env,
        ):
            # Run the agent with a curl trigger to test the MCP connection
            script = self.quickstart_dir / "app.py"
            result = run_quickstart_script(
                script,
                cwd=self.quickstart_dir,
                env=self.env,
                timeout=180,
                use_dapr=True,
                app_id="mcp-agent-streamable",
                app_port=8001,
                stream_logs=True,
                trigger_curl={
                    "url": "http://localhost:8001/run",
                    "method": "POST",
                    "data": {"task": "What is the weather in New York?"},
                    "headers": {"Content-Type": "application/json"},
                    "wait_seconds": 10,  # Give agent time to connect to MCP server and start
                },
            )

            assert result.returncode == 0, (
                f"Quickstart failed with return code {result.returncode}.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
            # expect some output
            assert len(result.stdout) > 0 or len(result.stderr) > 0
            # Verify MCP connection succeeded (should not see "Failed to load MCP tools" error)
            assert (
                "Failed to load MCP tools via streamable HTTP" not in result.stderr
            ), "MCP server connection failed. Check that server started correctly."
