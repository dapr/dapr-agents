"""Integration tests for 04-multi-agent-workflow-k8s example."""

import pytest


@pytest.mark.integration
class TestMultiAgentWorkflowK8sQuickstart:
    """Integration tests for 04-multi-agent-workflow-k8s example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir):
        """Setup test environment."""

    def test_multi_agent_workflow_k8s(self, dapr_runtime):  # noqa: ARG002
        pytest.skip(
            "Skipping 04-multi-agent-workflow-k8s test because it requires a Kubernetes cluster."
        )
