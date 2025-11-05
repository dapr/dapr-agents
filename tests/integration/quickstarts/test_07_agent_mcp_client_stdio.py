"""Integration tests for 07-agent-mcp-client-stdio quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestMCPClientStdioQuickstart:
    """Integration tests for 07-agent-mcp-client-stdio quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "07-agent-mcp-client-stdio"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_mcp_agent_stdio(self):
        """Test MCP agent with STDIO transport (agent.py)."""
        script = self.quickstart_dir / "agent.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=90,
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert "Available tools" in result.stdout or len(result.stdout) > 0
