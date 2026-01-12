"""Integration tests for 01-dapr-agents-fundamentals quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestHelloWorldQuickstart:
    """Integration tests for 01-dapr-agents-fundamentals quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "01-dapr-agents-fundamentals"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_01_llm_client(self, dapr_runtime):  # noqa: ARG002
        """Test LLM client example (01_llm_client.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "01_llm_client.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="llm-client",
            resources_path=self.quickstart_dir / "resources",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
        # Verify LLM response was received
        assert "Response:" in result.stdout

    def test_02_agent_llm(self, dapr_runtime):  # noqa: ARG002
        """Test agent with LLM example (02_agent_llm.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "02_agent_llm.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="agent-llm",
            resources_path=self.quickstart_dir / "resources",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
        # Verify agent responded
        assert "Agent:" in result.stdout or "weather" in result.stdout.lower()

    def test_03_agent_llm_tools(self, dapr_runtime):  # noqa: ARG002
        """Test agent with LLM and tools example (03_agent_llm_tools.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "03_agent_llm_tools.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="agent-llm",
            resources_path=self.quickstart_dir / "resources",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_04_agent_mcp_tools(self, dapr_runtime):  # noqa: ARG002
        """Test agent with MCP tools example (04_agent_mcp_tools.py)."""
        script_path = self.quickstart_dir / "04_agent_mcp_tools.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="agent-mcp",
            resources_path=self.quickstart_dir / "resources",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_05_agent_memory(self, dapr_runtime):  # noqa: ARG002
        """Test agent with memory example (05_agent_memory.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "05_agent_memory.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="agent-memory",
            resources_path=self.quickstart_dir / "resources",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
        # Verify agent responded and remembered the name
        assert "John" in result.stdout or "weather" in result.stdout.lower()

    def test_06_durable_agent_http(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent HTTP example (06_durable_agent_http.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "06_durable_agent_http.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="durable-agent",
            app_port=8001,
            resources_path=self.quickstart_dir / "resources",
            trigger_curl={
                "url": "http://localhost:8001/run",
                "method": "POST",
                "data": {"task": "What is the weather in London?"},
                "headers": {"Content-Type": "application/json"},
                "wait_seconds": 5,
            },
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_07_durable_agent_pubsub(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent pub/sub example (07_durable_agent_pubsub.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "07_durable_agent_pubsub.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="durable-agent-sub",
            dapr_http_port=3500,
            resources_path=self.quickstart_dir / "resources",
            trigger_pubsub={
                "pubsub_name": "message-pubsub",
                "topic": "weather.requests",
                "data": {"task": "What's the weather in Boston?"},
                "wait_seconds": 5,
            },
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_08_workflow_llm(self, dapr_runtime):  # noqa: ARG002
        """Test workflow with LLM activities example (08_workflow_llm.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "08_workflow_llm.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="workflow-llms",
            resources_path=self.quickstart_dir / "resources",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
        # Verify workflow completed successfully
        assert (
            "Workflow started:" in result.stdout
            or "Blog post" in result.stdout
            or "Final Blog Post" in result.stdout
        )

    def test_09_workflow_agents(self, dapr_runtime):  # noqa: ARG002
        """Test workflow with agent activities example (09_workflow_agents.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "09_workflow_agents.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="workflow-agents",
            resources_path=self.quickstart_dir / "resources",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
        # Verify workflow completed successfully
        assert (
            "Workflow started:" in result.stdout
            or "Recommendation" in result.stdout
            or "Final Recommendation" in result.stdout
        )

    def test_10_durable_agent_tracing(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent tracing example (10_durable_agent_tracing.py)."""
        script_path = self.quickstart_dir / "10_durable_agent_tracing.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="durable-agent-trace",
            resources_path=self.quickstart_dir / "resources",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
