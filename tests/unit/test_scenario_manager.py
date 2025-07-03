"""Unit tests for scenario manager."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.utils import ScenarioManager, DevelopmentScenario


class TestScenarioManager:
    """Test cases for ScenarioManager."""

    def test_init(self):
        """Test ScenarioManager initialization."""
        manager = ScenarioManager()

        assert manager.workspace_root.name == "dapr-agents"
        assert manager.dapr_root.name == "dapr"
        assert manager.python_sdk_root.name == "python-sdk"
        assert manager.components_contrib_root.name == "components-contrib"

    def test_detect_scenario_local_full(self):
        """Test detection of local full development scenario."""
        manager = ScenarioManager()

        # Mock all repos as being in development
        with patch.object(manager, "_detect_development_repos") as mock_detect:
            mock_detect.return_value = {
                "dapr": True,
                "python_sdk": True,
                "components_contrib": True,
                "dapr_agents": True,
            }

            scenario = manager.detect_scenario()
            assert scenario == DevelopmentScenario.LOCAL_FULL

    def test_detect_scenario_local_partial(self):
        """Test detection of local partial development scenario."""
        manager = ScenarioManager()

        # Mock only Python SDK and agents in development
        with patch.object(manager, "_detect_development_repos") as mock_detect:
            mock_detect.return_value = {
                "dapr": False,
                "python_sdk": True,
                "components_contrib": False,
                "dapr_agents": True,
            }

            scenario = manager.detect_scenario()
            assert scenario == DevelopmentScenario.LOCAL_PARTIAL

    def test_detect_scenario_agent_only(self):
        """Test detection of agent-only development scenario."""
        manager = ScenarioManager()

        # Mock only agents in development
        with patch.object(manager, "_detect_development_repos") as mock_detect:
            mock_detect.return_value = {
                "dapr": False,
                "python_sdk": False,
                "components_contrib": False,
                "dapr_agents": True,
            }

            scenario = manager.detect_scenario()
            assert scenario == DevelopmentScenario.AGENT_ONLY

    def test_get_component_config_dir(self):
        """Test getting component configuration directory for scenarios."""
        manager = ScenarioManager()

        # Test each scenario
        local_full_dir = manager.get_component_config_dir(
            DevelopmentScenario.LOCAL_FULL
        )
        assert "local_dev" in local_full_dir

        local_partial_dir = manager.get_component_config_dir(
            DevelopmentScenario.LOCAL_PARTIAL
        )
        assert "partial_dev" in local_partial_dir

        agent_only_dir = manager.get_component_config_dir(
            DevelopmentScenario.AGENT_ONLY
        )
        assert "production" in agent_only_dir

        production_dir = manager.get_component_config_dir(
            DevelopmentScenario.PRODUCTION
        )
        assert "production" in production_dir

    def test_get_dapr_endpoints(self):
        """Test getting Dapr endpoints."""
        manager = ScenarioManager()

        endpoints = manager.get_dapr_endpoints(DevelopmentScenario.LOCAL_FULL)

        assert "http" in endpoints
        assert "grpc" in endpoints
        assert endpoints["http"] == "http://localhost:3500"
        assert endpoints["grpc"] == "localhost:50001"

    def test_get_scenario_description(self):
        """Test getting scenario descriptions."""
        manager = ScenarioManager()

        # Test each scenario has a description
        for scenario in DevelopmentScenario:
            description = manager.get_scenario_description(scenario)
            assert isinstance(description, str)
            assert len(description) > 0
            # Check that key words from scenario are in description
            scenario_words = scenario.value.replace("_", " ").split()
            description_lower = description.lower()
            assert any(word in description_lower for word in scenario_words)
