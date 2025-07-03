"""Development scenario detection and management."""

from pathlib import Path
from enum import Enum
from typing import Dict
import subprocess
import logging

logger = logging.getLogger(__name__)


class DevelopmentScenario(Enum):
    """Development scenarios for testing."""

    LOCAL_FULL = "local_full"  # All repos in development
    LOCAL_PARTIAL = "local_partial"  # SDK + agents in development
    AGENT_ONLY = "agent_only"  # Only agents in development
    PRODUCTION = "production"  # All released versions


class ScenarioManager:
    """Manages detection and configuration of development scenarios."""

    def __init__(self):
        self.workspace_root = Path(__file__).parent.parent.parent
        self.dapr_root = self.workspace_root.parent / "dapr"
        self.python_sdk_root = self.workspace_root.parent / "python-sdk"
        self.components_contrib_root = self.workspace_root.parent / "components-contrib"

    def detect_scenario(self) -> DevelopmentScenario:
        """Auto-detect current development scenario based on repository states."""
        development_repos = self._detect_development_repos()

        if all(development_repos.values()):
            return DevelopmentScenario.LOCAL_FULL
        elif development_repos.get("python_sdk", False):
            return DevelopmentScenario.LOCAL_PARTIAL
        elif all(
            not development_repos[repo]
            for repo in ["dapr", "python_sdk", "components_contrib"]
        ):
            return DevelopmentScenario.AGENT_ONLY
        else:
            # Mixed scenario, default to production for safety
            return DevelopmentScenario.PRODUCTION

    def _detect_development_repos(self) -> Dict[str, bool]:
        """Detect which repositories are in local development mode."""
        return {
            "dapr": self._is_local_development(self.dapr_root),
            "python_sdk": self._is_local_development(self.python_sdk_root),
            "components_contrib": self._is_local_development(
                self.components_contrib_root
            ),
            "dapr_agents": True,  # Always local since we're in this repo
        }

    def _is_local_development(self, repo_path: Path) -> bool:
        """Check if a repository is in local development mode."""
        if not repo_path.exists():
            return False

        # Check if it's a git repository with recent commits
        try:
            git_dir = repo_path / ".git"
            if not git_dir.exists():
                return False

            # Check for recent commits (within last 30 days as indicator of active development)
            result = subprocess.run(
                ["git", "log", "--since=30.days.ago", "--oneline"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )

            # If there are recent commits, consider it active development
            return len(result.stdout.strip()) > 0

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            # If we can't check git status, assume not in development
            return False

    def get_component_config_dir(self, scenario: DevelopmentScenario) -> str:
        """Get appropriate component configuration directory for scenario."""
        scenario_map = {
            DevelopmentScenario.LOCAL_FULL: "tests/components/local_dev",
            DevelopmentScenario.LOCAL_PARTIAL: "tests/components/partial_dev",
            DevelopmentScenario.AGENT_ONLY: "tests/components/production",
            DevelopmentScenario.PRODUCTION: "tests/components/production",
        }
        return scenario_map[scenario]

    def get_dapr_endpoints(self, scenario: DevelopmentScenario) -> Dict[str, str]:
        """Get Dapr endpoints for scenario."""
        # For now, all scenarios use the same endpoints
        # This could be extended to use different ports for different scenarios
        return {"http": "http://localhost:3500", "grpc": "localhost:50001"}

    def validate_scenario_requirements(
        self, scenario: DevelopmentScenario
    ) -> Dict[str, bool]:
        """Validate that scenario requirements are met."""
        requirements = {
            "dapr_runtime": self._check_dapr_runtime(),
            "python_sdk": self._check_python_sdk(),
            "components": self._check_components_available(scenario),
        }

        if scenario == DevelopmentScenario.LOCAL_FULL:
            requirements.update(
                {
                    "local_dapr_build": self._check_local_dapr_build(),
                    "local_sdk_build": self._check_local_sdk_build(),
                    "local_components_build": self._check_local_components_build(),
                }
            )

        return requirements

    def _check_dapr_runtime(self) -> bool:
        """Check if Dapr runtime is available."""
        try:
            result = subprocess.run(
                ["dapr", "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return False

    def _check_python_sdk(self) -> bool:
        """Check if Python SDK is available."""
        try:
            import dapr

            return True
        except ImportError:
            return False

    def _check_components_available(self, scenario: DevelopmentScenario) -> bool:
        """Check if required components are available for scenario."""
        component_dir = Path(self.get_component_config_dir(scenario))
        return component_dir.exists() and any(component_dir.glob("*.yaml"))

    def _check_local_dapr_build(self) -> bool:
        """Check if local Dapr build is available."""
        if not self.dapr_root.exists():
            return False

        # Check for built binaries
        dapr_binary = self.dapr_root / "dist" / "darwin_amd64" / "release" / "dapr"
        if not dapr_binary.exists():
            # Try alternate location
            dapr_binary = self.dapr_root / "bin" / "dapr"

        return dapr_binary.exists()

    def _check_local_sdk_build(self) -> bool:
        """Check if local Python SDK build is available."""
        if not self.python_sdk_root.exists():
            return False

        # Check for setup.py or pyproject.toml
        return (self.python_sdk_root / "setup.py").exists() or (
            self.python_sdk_root / "pyproject.toml"
        ).exists()

    def _check_local_components_build(self) -> bool:
        """Check if local components-contrib build is available."""
        if not self.components_contrib_root.exists():
            return False

        # Check for built components
        components_dir = self.components_contrib_root / "dist"
        return components_dir.exists() and any(components_dir.iterdir())

    def get_scenario_description(self, scenario: DevelopmentScenario) -> str:
        """Get human-readable description of scenario."""
        descriptions = {
            DevelopmentScenario.LOCAL_FULL: "Full local development - all repositories (dapr, python-sdk, components-contrib, dapr-agents) in development",
            DevelopmentScenario.LOCAL_PARTIAL: "Partial local development - python-sdk and dapr-agents in development, others released",
            DevelopmentScenario.AGENT_ONLY: "Agent-only development - only dapr-agents in development, all dependencies released",
            DevelopmentScenario.PRODUCTION: "Production-like testing - all dependencies use released versions",
        }
        return descriptions[scenario]


if __name__ == "__main__":
    # CLI interface for scenario detection
    import argparse

    parser = argparse.ArgumentParser(description="Detect development scenario")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    manager = ScenarioManager()
    scenario = manager.detect_scenario()

    print(f"Detected scenario: {scenario.value}")
    print(f"Description: {manager.get_scenario_description(scenario)}")

    if args.verbose:
        print("\nRepository states:")
        repos = manager._detect_development_repos()
        for repo, is_dev in repos.items():
            status = "LOCAL DEVELOPMENT" if is_dev else "RELEASED/STABLE"
            print(f"  {repo}: {status}")

        print("\nRequirements validation:")
        requirements = manager.validate_scenario_requirements(scenario)
        for req, met in requirements.items():
            status = "✓" if met else "✗"
            print(f"  {status} {req}")
