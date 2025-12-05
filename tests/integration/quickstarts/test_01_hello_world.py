"""Integration tests for 01-hello-world quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestHelloWorldQuickstart:
    """Integration tests for 01-hello-world quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "01-hello-world"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_agent_with_memory(self, dapr_runtime):  # noqa: ARG002
        """Test agent with memory example (01_agent_with_memory.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "01_agent_with_memory.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=120,
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

    def test_agent_with_durable_execution(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent execution example (02_agent_with_durable_execution.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "02_agent_with_durable_execution.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=120,
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

    def test_workflow_with_llms(self, dapr_runtime):  # noqa: ARG002
        """Test workflow with LLM activities example (03_workflow_with_llms.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "03_workflow_with_llms.py"
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

    def test_workflow_with_agents(self, dapr_runtime):  # noqa: ARG002
        """Test workflow with agent activities example (04_workflow_with_agents.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "04_workflow_with_agents.py"
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
