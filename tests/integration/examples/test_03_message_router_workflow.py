#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Integration tests for 03-message-router-workflow example."""

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_script


@pytest.mark.integration
class TestMessageRouterWorkflowQuickstart:
    """Integration tests for 03-message-router-workflow example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = examples_dir / "03-message-router-workflow"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_message_router_workflow(self, dapr_runtime):  # noqa: ARG002
        """Test message router workflow (app.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.

        This test starts the workflow app and triggers it by publishing a message
        to the blog.requests topic, which will start the blog_workflow.
        """
        script = self.quickstart_dir / "app.py"
        result = run_quickstart_or_examples_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="message-router-app",
            dapr_http_port=3500,
            trigger_pubsub={
                "pubsub_name": "messagepubsub",
                "topic": "blog.requests",
                "data": {"topic": "AI Agents"},
                "wait_seconds": 5,
            },
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0
