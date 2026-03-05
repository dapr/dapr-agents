"""Integration tests for 04-multi-agent-workflows example."""

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_multi_app


@pytest.mark.integration
class TestMultiAgentWorkflowsQuickstart:
    """Integration tests for 04-multi-agent-workflows example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = examples_dir / "04-multi-agent-workflows"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_random_orchestrator(self, dapr_runtime):  # noqa: ARG002
        # Use a different registry store to isolate from other orchestrators (e.g., random)
        # This prevents the round-robin orchestrator from selecting other orchestrators
        # as agents when they're registered in the same team registry.
        # Note: Agents still use team "fellowship" but with isolated registry store.
        test_env = {**self.env, "REGISTRY_STATE_STORE": "agentregistrystore_random"}
        test_env["DAPR_HOST_IP"] = "127.0.0.1"
        dapr_yaml = self.quickstart_dir / "dapr-random.yaml"
        result = run_quickstart_or_examples_multi_app(
            dapr_yaml,
            cwd=self.quickstart_dir,
            env=test_env,
            timeout=300,
            stream_logs=True,
        )

        assert result.returncode == 0, (
            f"Multi-app run failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_roundrobin_orchestrator(self, dapr_runtime):  # noqa: ARG002
        dapr_yaml = self.quickstart_dir / "dapr-roundrobin.yaml"
        # Use a different registry store to isolate from other orchestrators (e.g., random)
        # This prevents the round-robin orchestrator from selecting other orchestrators
        # as agents when they're registered in the same team registry.
        # Note: Agents still use team "fellowship" but with isolated registry store.
        test_env = {**self.env, "REGISTRY_STATE_STORE": "agentregistrystore_roundrobin"}
        test_env["DAPR_HOST_IP"] = "127.0.0.1"
        result = run_quickstart_or_examples_multi_app(
            dapr_yaml,
            cwd=self.quickstart_dir,
            env=test_env,
            timeout=300,
            stream_logs=True,
        )

        assert result.returncode == 0, (
            f"Multi-app run failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_agent_orchestrator(self, dapr_runtime):  # noqa: ARG002
        # Use a different registry store to isolate from other orchestrators (e.g., random)
        # This prevents the round-robin orchestrator from selecting other orchestrators
        # as agents when they're registered in the same team registry.
        # Note: Agents still use team "fellowship" but with isolated registry store.
        test_env = {**self.env, "REGISTRY_STATE_STORE": "agentregistrystore_agent"}
        test_env["DAPR_HOST_IP"] = "127.0.0.1"
        dapr_yaml = self.quickstart_dir / "dapr-agent.yaml"
        result = run_quickstart_or_examples_multi_app(
            dapr_yaml,
            cwd=self.quickstart_dir,
            env=test_env,
            timeout=300,
            stream_logs=True,
        )

        assert result.returncode == 0, (
            f"Multi-app run failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0
