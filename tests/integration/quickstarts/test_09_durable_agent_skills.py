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

"""Integration test for the 09_durable_agent_skills quickstart."""

import os

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_script


@pytest.mark.integration
class TestDurableAgentSkillsQuickstart:
    """Integration tests for the durable agent filesystem skills quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key, ollama_resources_dir, is_ollama):
        """Set up the quickstart directory and model configuration."""
        self.quickstart_dir = quickstarts_dir
        self.env = {"OPENAI_API_KEY": openai_api_key}
        self.resources_path = ollama_resources_dir
        if is_ollama:
            self.env["OLLAMA_MODEL"] = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

    def test_09_durable_agent_skills(self, dapr_runtime):  # noqa: ARG002
        """Run a delayed-shipment task through the filesystem skills quickstart."""
        script_path = self.quickstart_dir / "09_durable_agent_skills.py"
        result = run_quickstart_or_examples_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=300,
            use_dapr=True,
            app_id="durable-agent-skills",
            app_port=8001,
            resources_path=self.resources_path,
            trigger_curl={
                # Stream until completion; the test helper terminates the server after the response
                "url": "http://localhost:8001/agent/run/stream",
                "method": "POST",
                "data": {
                    "task": (
                        "Shipment SHP-042 has had no carrier scan for 36 hours and is stalled "
                        "in transit. It may miss its promised delivery window. Use the "
                        "shipment-exceptions skill and read "
                        "references/exception-routing-policy.md before recommending "
                        "the appropriate route."
                    ),
                    "include_complete_message": True,
                },
                "headers": {
                    "Accept": "application/x-ndjson",
                    "Content-Type": "application/json",
                },
                "wait_seconds": 10,
                "request_timeout": 300,
            },
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code "
            f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        combined = f"{result.stdout}\n{result.stderr}".lower()

        # Tool metadata and route terms confirm the policy resource was loaded and decoded
        assert "read_skill_resource" in combined
        assert "references/exception-routing-policy.md" in combined
        assert '"encoding": "utf-8"' in combined

        for route in (
            "carrier escalation",
            "warehouse review",
            "customer notification",
            "manual review",
        ):
            assert route in combined
